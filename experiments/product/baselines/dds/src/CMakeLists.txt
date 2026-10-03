cmake_minimum_required(VERSION 3.16)
project(sentinel_secure_dds LANGUAGES C)

find_package(CycloneDDS REQUIRED)
idlc_generate(TARGET sentinel_action_type FILES SentinelAction.idl WARNINGS no-implicit-extensibility)

add_executable(sentinel_dds_endpoint endpoint.c)
target_link_libraries(sentinel_dds_endpoint sentinel_action_type CycloneDDS::ddsc)
target_compile_features(sentinel_dds_endpoint PRIVATE c_std_11)
