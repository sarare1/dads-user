# Data handling rule

This rule says which data may be put into which DADS installation. It applies to everyone who uploads,
generates, copies or exports data with DADS, and to every copy of a trained model.

## Why it is needed
DADS learns from pulse descriptor words (PDWs): frequency, PRI, pulse width and similar emitter
parameters. Synthetic tracks from the built-in generator are harmless. Parameters measured from real
emitters (ELINT) usually are not. They can be classified, export-controlled (for example under ITAR or
the EU dual-use regulation) or owned by a third party under contract. A model trained on such data carries
that sensitivity with it: its prototypes and thresholds encode the parameters it learned.

DADS itself does not encrypt data at rest, and the public server is a commercial VPS. So the protection has
to come from where the data is allowed to go.

## Data classes

| Class | What it is | Examples |
|---|---|---|
| **A: Synthetic** | Made by the DADS generator or another simulator from invented parameters | `generated_*` datasets, Live Simulation tracks |
| **B: Public** | Real-world values taken only from open publications, with the source recorded | Parameter ranges from open literature or datasheets |
| **C: Restricted** | Anything measured from real emitters, supplied by a partner, or marked with a protective marking or export-control notice | Recorded ELINT/PDW files, partner datasets, trial data |

If you are unsure, treat the data as **class C** until whoever supplied it confirms otherwise in writing.

## Where each class may go

| Installation | A | B | C |
|---|---|---|---|
| Public server (Hostinger VPS, branch `main`) | ✅ | ✅ | ❌ never |
| Personal or lab computer (branch `main_local`) | ✅ | ✅ | ❌ unless that computer is accredited for the data |
| Accredited or air-gapped machine, set up under the data owner's rules | ✅ | ✅ | ✅ |

Rules for class C data:
1. Never upload it to the public server, not even briefly and not "only for a test". Deleting it later
   does not undo the breach, because backups and logs keep copies.
2. Models trained on class C data are class C too. The same goes for their downloaded packages, test
   results, reports and screenshots of them. They stay on the accredited machine.
3. Move class C data only on media and over channels the data owner has approved. Do not use email,
   cloud drives or chat tools.
4. Record each class C dataset in that machine's data register: source, marking, owner, date received
   and date destroyed. DADS stores each dataset's SHA-256 fingerprint on the Dataset page; use it as the
   register's identifier.

## For the thesis and publications
Publish results from class A or B data only, or results the data owner has cleared in writing. This
includes confusion matrices, prototypes and per-class tables. Aggregated numbers can still reveal
parameters.

## What DADS does to help
- Each dataset and each run records the SHA-256 of its data, so you can prove which data trained a model.
- Downloaded model packages carry a manifest with their provenance, and the server signs them.
- The audit log (`data/audit.log`) records who uploaded, trained, deleted and downloaded what, and from where.
- Sign-in is required for everything. Accounts are personal, so give each person their own account
  (`scripts/manage_users.py add <name>`).

None of this makes the public server fit for class C data. That needs an accredited environment.

## If class C data reached the wrong place
Stop using the installation. Do not delete anything yet, because the audit log and backups show what
happened. Tell the data owner and your supervisor or security officer the same day, and follow their
instructions for cleaning up.
