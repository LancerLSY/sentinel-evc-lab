# Sentinel EVC project website

Self-contained static project page for Vercel.

- Repository directory: `project-site`
- Framework preset: Other
- Output directory: `public`
- Build and install commands: disabled
- Project path: `/research/sentinel-vla/project/`
- No Sites runtime, database, account or secret is required to serve the project page.

All project scripts, fonts, diagrams, videos and recorded 3D assets are included.
Other paths are reverse-proxied to the existing personal site, preserving its content
and APIs while the project page is hosted independently. Missing project assets
are not proxied to the old host.

Import this Git branch in Vercel and set the Root Directory to `project-site`.
Use this branch as the production branch. Verify the project page, native replay,
video seek/range requests and personal-site fallback on the deployment URL before
changing DNS for `lansiyao.com`. Obtain the required DNS records from the actual
Vercel project domain settings; do not replace MX or mail-related TXT records.
Retain the existing hosting and DNS records until the new deployment is verified.

`PAGE_SHA256SUMS` records the exact published static files. Website deployment does
not modify experiment results, model weights, controller contracts or personal
site source. Upstream font and model notices are retained with their assets.
