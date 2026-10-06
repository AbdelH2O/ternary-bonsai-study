# Pause and resume the chat experiment

The experiment can be paused and resumed later, including after a host reboot. State is stored in the repository's `work/qat08_chat` and `results/qat08_chat`, rather than in the chat session or a transient systemd unit.

From the repository root:

```bash
.venv/bin/python analysis/bonsai2/replication/qat08_chat_control.py status
.venv/bin/python analysis/bonsai2/replication/qat08_chat_control.py pause
.venv/bin/python analysis/bonsai2/replication/qat08_chat_control.py resume
```

You can also tell the coding agent **“pause the Bonsai chat experiment”** or **“resume the Bonsai chat experiment.”** These commands use the exact `qat08-chat-run` systemd user unit; there is no process-name pattern killing.

After requesting a pause, wait until `status` reports `state: paused` and the unit is inactive before restarting the machine. The pause takes effect at a safe boundary. During teacher generation, at most eight in-flight requests drain; no more requests begin. During training, the current optimizer step finishes and a fresh `resume.pt` is saved before the next step begins. A running evaluation shard may finish before the pause takes effect.

After a reboot, run `resume` whenever convenient. It recreates the systemd unit, verifies frozen input/code/environment hashes, and continues the unfinished stage. A paused experiment does **not** automatically restart at boot or login.

| Stage | Durable state | Replay after interruption |
|---|---|---|
| Teacher generation | Individual response records, flushed and synced to disk | Unrecorded in-flight responses |
| Packing / protocol freeze | Deterministic intermediate files; atomic publication | Unsealed packing or contamination audit can be recomputed |
| Student training | Latent weights, BF16 optimizer moments, optimizer step/data offset, RNG states | None of the completed steps after a clean pause; up to about 20 minutes after sudden power loss |
| Export | Complete GGUF and matching export manifest | An unfinished export is preserved and rebuilt |
| Evaluation | Completed scorer shards and GSM8K response records | Unfinished shard or unrecorded response |
| Reporting | Atomic report/document replacement | Safe to rerun |

The original 20-minute periodic checkpoint cadence is unchanged. A **clean pause creates an extra checkpoint immediately at a completed-step boundary**. For an unexpected power failure or forced kill, the last periodic checkpoint remains intact; `resume` may replay later steps. A truncated final JSON record is recoverable and the original file is preserved; corruption in the middle of a journal stops execution for investigation.

The controller tracks cumulative recorded stage time and stops at the frozen compute ceiling. Rebooting does not grant a new training budget. If a hard interruption leaves no finish event, the report flags the timing gap rather than claiming an exact duration.

Verified before launch: five steps of the original trainer on a tiny CPU fixture were bit-for-bit identical to two steps, a fresh-process restart, and three more steps in weights, BF16 moments, losses, optimizer count and RNG state. Journal-tail recovery and refusal to overwrite a freeze were also tested. This tests restart mechanics without training the research model. No physical reboot of the user's machine was performed.

The original scientific design and scripts are untouched. Added execution controls and test evidence are pinned in [resume_amendment_v1.json](results/qat08_chat/resume_amendment_v1.json). The separate controller handles generation, final protocol freeze, training, export, evaluation and the requested report/status updates.

Status/log files:

- `work/qat08_chat/run_state.json`: current stage and running/paused/failed/complete state.
- `work/qat08_chat/run.log`: persistent systemd job output.
- `work/qat08_chat/execution.jsonl`: stage timing, pause and artifact-recovery incidents.
- `work/qat08_chat/qat_chat_42/resume.pt`: training restart state.
- `work/qat08_chat/PAUSE_REQUESTED`: persistent pause marker; explicit `resume` clears it.
