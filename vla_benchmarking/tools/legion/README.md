# Legion support tools

`Export-LegionCanaryArtifacts.ps1` copies completed canary artifacts into a
hash-verified local archive. The project-specific export logic lives here so it
is available in a clean clone.

Credentials remain outside the project code. By default the exporter uses the
local helpers in `.codex/legion-local` at the repository root:

- `Invoke-Legion.ps1`
- `legion-askpass.cmd`

Pass `-LegionHelperRoot` to use an equivalent local helper directory elsewhere.
