#include "dds/dds.h"
#include "SentinelAction.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LINE (1024 * 1024)

static int fail_entity(const char *operation, dds_entity_t entity)
{
  fprintf(stderr, "ERROR\t%s\t%d\t%s\n", operation, (int)entity,
          dds_strretcode(-entity));
  return 2;
}

static int64_t monotonic_ns(void)
{
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
    return -1;
  }
  return ((int64_t)ts.tv_sec * INT64_C(1000000000)) + ts.tv_nsec;
}

static dds_entity_t create_topic(dds_entity_t participant, const char *name)
{
  return dds_create_topic(participant, &Sentinel_ActionEnvelope_desc, name, NULL, NULL);
}

static int wait_for_match(dds_entity_t endpoint, uint32_t mask, int timeout_ms)
{
  int elapsed = 0;
  dds_return_t rc = dds_set_status_mask(endpoint, mask);
  if (rc != DDS_RETCODE_OK) {
    fprintf(stderr, "ERROR\tset_status_mask\t%d\t%s\n", (int)rc,
            dds_strretcode(-rc));
    return 2;
  }
  while (elapsed < timeout_ms) {
    uint32_t status = 0;
    rc = dds_get_status_changes(endpoint, &status);
    if (rc != DDS_RETCODE_OK) {
      fprintf(stderr, "ERROR\tget_status_changes\t%d\t%s\n", (int)rc,
              dds_strretcode(-rc));
      return 2;
    }
    if ((status & mask) != 0) {
      return 0;
    }
    dds_sleepfor(DDS_MSECS(10));
    elapsed += 10;
  }
  fprintf(stderr, "ERROR\tmatch_timeout\t%d\tno matched secure peer\n", timeout_ms);
  return 3;
}

static int publish_file(const char *topic_name, const char *path, int timeout_ms)
{
  int result = 0;
  FILE *input = NULL;
  char *line = NULL;
  dds_entity_t participant = dds_create_participant(DDS_DOMAIN_DEFAULT, NULL, NULL);
  if (participant < 0) return fail_entity("create_participant", participant);

  dds_entity_t topic = create_topic(participant, topic_name);
  if (topic < 0) { result = fail_entity("create_topic", topic); goto done; }

  dds_qos_t *qos = dds_create_qos();
  dds_qset_reliability(qos, DDS_RELIABILITY_RELIABLE, DDS_SECS(5));
  dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 128);
  dds_entity_t writer = dds_create_writer(participant, topic, qos, NULL);
  dds_delete_qos(qos);
  if (writer < 0) { result = fail_entity("create_writer", writer); goto done; }

  printf("READY\tpublisher\t%s\n", topic_name);
  fflush(stdout);
  result = wait_for_match(writer, DDS_PUBLICATION_MATCHED_STATUS, timeout_ms);
  if (result != 0) goto done;
  printf("MATCHED\tpublisher\t%s\t%" PRId64 "\n", topic_name, monotonic_ns());
  fflush(stdout);

  input = fopen(path, "rb");
  if (input == NULL) {
    fprintf(stderr, "ERROR\topen_input\t%d\t%s\n", errno, strerror(errno));
    result = 2;
    goto done;
  }
  line = malloc(MAX_LINE);
  if (line == NULL) { result = 2; goto done; }

  size_t count = 0;
  while (fgets(line, MAX_LINE, input) != NULL) {
    size_t length = strlen(line);
    while (length > 0 && (line[length - 1] == '\n' || line[length - 1] == '\r')) {
      line[--length] = '\0';
    }
    if (length == 0) continue;
    Sentinel_ActionEnvelope message = {.payload = line};
    dds_return_t rc = dds_write(writer, &message);
    if (rc != DDS_RETCODE_OK) {
      fprintf(stderr, "ERROR\twrite\t%d\t%s\n", (int)rc, dds_strretcode(-rc));
      result = 2;
      goto done;
    }
    printf("SENT\t%zu\t%" PRId64 "\n", count, monotonic_ns());
    fflush(stdout);
    ++count;
  }
  printf("COMPLETE\tpublisher\t%zu\n", count);
  fflush(stdout);

done:
  free(line);
  if (input != NULL) fclose(input);
  dds_delete(participant);
  return result;
}

static int subscribe_count(const char *topic_name, size_t expected, int timeout_ms)
{
  int result = 0;
  dds_entity_t participant = dds_create_participant(DDS_DOMAIN_DEFAULT, NULL, NULL);
  if (participant < 0) return fail_entity("create_participant", participant);

  dds_entity_t topic = create_topic(participant, topic_name);
  if (topic < 0) { result = fail_entity("create_topic", topic); goto done; }

  dds_qos_t *qos = dds_create_qos();
  dds_qset_reliability(qos, DDS_RELIABILITY_RELIABLE, DDS_SECS(5));
  dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 128);
  dds_entity_t reader = dds_create_reader(participant, topic, qos, NULL);
  dds_delete_qos(qos);
  if (reader < 0) { result = fail_entity("create_reader", reader); goto done; }

  printf("READY\tsubscriber\t%s\n", topic_name);
  fflush(stdout);
  Sentinel_ActionEnvelope *sample = Sentinel_ActionEnvelope__alloc();
  void *samples[1] = {sample};
  dds_sample_info_t info[1];
  size_t received = 0;
  int elapsed = 0;
  bool matched_logged = false;
  dds_set_status_mask(reader, DDS_SUBSCRIPTION_MATCHED_STATUS);

  while (received < expected && elapsed < timeout_ms) {
    uint32_t status = 0;
    if (!matched_logged && dds_get_status_changes(reader, &status) == DDS_RETCODE_OK &&
        (status & DDS_SUBSCRIPTION_MATCHED_STATUS) != 0) {
      printf("MATCHED\tsubscriber\t%s\t%" PRId64 "\n", topic_name, monotonic_ns());
      fflush(stdout);
      matched_logged = true;
    }
    dds_return_t rc = dds_take(reader, samples, info, 1, 1);
    if (rc < 0) {
      fprintf(stderr, "ERROR\ttake\t%d\t%s\n", (int)rc, dds_strretcode(-rc));
      result = 2;
      break;
    }
    if (rc > 0 && info[0].valid_data) {
      printf("RECEIVED\t%zu\t%" PRId64 "\t%s\n", received, monotonic_ns(), sample->payload);
      fflush(stdout);
      ++received;
    }
    dds_sleepfor(DDS_MSECS(5));
    elapsed += 5;
  }
  if (received != expected) {
    fprintf(stderr, "ERROR\treceive_timeout\t%zu\t%zu\n", received, expected);
    result = 3;
  } else {
    printf("COMPLETE\tsubscriber\t%zu\n", received);
    fflush(stdout);
  }
  Sentinel_ActionEnvelope_free(sample, DDS_FREE_ALL);

done:
  dds_delete(participant);
  return result;
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    fprintf(stderr, "usage: %s publish TOPIC NDJSON [TIMEOUT_MS] | subscribe TOPIC COUNT [TIMEOUT_MS]\n", argv[0]);
    return 64;
  }
  const int timeout_ms = argc >= 5 ? atoi(argv[4]) : 15000;
  if (strcmp(argv[1], "publish") == 0 && argc >= 4) {
    return publish_file(argv[2], argv[3], timeout_ms);
  }
  if (strcmp(argv[1], "subscribe") == 0 && argc >= 4) {
    return subscribe_count(argv[2], (size_t)strtoull(argv[3], NULL, 10), timeout_ms);
  }
  fprintf(stderr, "invalid arguments\n");
  return 64;
}
