# Integrity hashes

SHA-256 of the artefacts a reader is most likely to want to check. Values recorded in the
annotated tag `prereg-lock-v1` and in the research log are reproduced in
`registration/REGISTRATION_RECORD.md`, which also explains why the pre-registration document
has two legitimate hashes (line endings).

The Zenodo data deposit v3 (10.5281/zenodo.22705927) has md5 `aa3228d3ae40f759fc0f741550e9596c`,
sha256 `d377e8f827843d4240e5a0926e5d7428e30ac6ab00a25563406d138e00754fcf`.

| file | sha256 | what it is |
|---|---|---|
| `registration/lock_bundle_prereg_v1.zip` | `06beb77fb131d2cdcccafb0bb5ce6800532ba5e7be9c88ffd16349607cb6f71c` | the lock bundle: the canonical identifier of the registration |
| `registration/prereg_test_blocks_v1_LOCKED.md` | `9fc24cc719df69643d5a73f0881349be9646e04ff5c4339f6573ae8d88de7c5e` | the pre-registration (LF in git; the recorded CRLF value is 9fc24cc7…7c5e) |
| `registration/test_event_list_blockA_LOCKED.csv` | `b339f1795aaf977ebef0a2e8110d5e4df3c98e0cea0f902f6ec22a4f658b05e3` | the frozen 2025 test-block list, 46 rows |
| `registration/PREREG_AMENDMENT_9_2026-08-28.md` | `241636d8fe1002f274dcceffd0e6975c56e831a5377b288d80079abd102ae00f` | the amendment, written before registration, inside the same bundle |
| `scripts/test_access_v1.py` | `6b403afc53b41a4f0c5bf75b45954ae39cdd35b06ee9d2ca6feef56ee897b1c4` | the frozen scoring script (REV 4) used at the single access |
| `results/access_2025/access_20260828T123059Z_per_event.csv` | `2c1dd44498b3fe76be7005b493eb0d2424c8347d601efcca1a505418fceb5e79` | the locked per-event result table of the single access |
| `results/access_2025/access_20260828T123059Z_results.json` | `9f3c1d8410ccb79f2e1841ed2e04f191152d818f8e216f8dc404e7e3b8262226` | the primary and secondary test outcomes |
| `results/access_2025/access_console_20260828.log` | `802ba2ccbab8765a26d2ad6f70de4547b4c583c6f2f3b727021f8853ba574eee` | the console log of the access: environment pin, input-hash gate, checkpoint verification |
| `results/exploratory_2026/exploratory_blockB_20260908T140106Z_per_event.csv` | `0ae9318865a31d90fd0050e7d4eb4c16ad845cd6c1d9cc146b4f974ff71d598b` | 2026 block, exploratory |
| `results/encoder_study/master_results.json` | `3e81e66b7081b7c8add0db9b3ed0a5b2d2035455cd4c7d317f9fb51edc3f7757` | six encoders over the 77 fires, exploratory |
| `results/number_register.json` | `12762d8696e4b567488869401f9252d3d226f2db5f92bc993ec61a9819dd0d75` | every number the letter prints, with its source |

## Every file in this repository

Regenerate with:

```bash
find . -type f -not -path './.git/*' | sort | xargs sha256sum
```

| file | bytes | sha256 |
|---|---|---|
| `.gitignore` | 506 | `ea754e398cf2f1281a2053bbec9c2e08cf9bcc2ed82de62d167f481e4808de11` |
| `.zenodo.json` | 3256 | `3001c0b7cfda5c8ebcdd5e29f90a5afe8cc0dc81f88f45deb982da487a4e5adc` |
| `catalogue/dataset_v6_1_manifest.csv` | 1836457 | `cdff3201039e1846530568839c71254241a0c19881bf33b0bfff1bd1c367e1df` |
| `catalogue/dev_curation_list_2026-08-25.csv` | 54641 | `3e0bc88c11dd626c85c1f1b164585bded25bea192a95303c3f92d0d4b2be6878` |
| `catalogue/dev_curation_selected_2026-08-25.csv` | 11820 | `68e3aea2b31a9e05192325c505eb5fb78f309dbf5b9c87b927030e47cd7a129f` |
| `catalogue/event_catalogue_162_export_2026-09-11.csv` | 49260 | `4c51db5afaca7164e10cef2ff2cdbc7a1344c8fddf8fd07839612fa5c39437ca` |
| `catalogue/EXCLUSIONS_E2_2026-08-25.md` | 4502 | `ecf11e5ed45c3de70c606ea78a90fccd3499b888a044bfe142c0854f5afa8b0b` |
| `catalogue/FREEZE_REPORT_2026-08-25.md` | 6871 | `030cebc746cbc450db3c6be3dbed82bdeba13bff8265d8d9e489daa8221aaf17` |
| `catalogue/frozen_dev_pool_2026-08-25.csv` | 7061 | `1ff9eea16a46efeb93cfacbb91e83b949669a9b29d9a2c871e2c1daeb412714c` |
| `catalogue/gate_results_2026-08-24.csv` | 7002 | `e20f96184baa0fc01796bf6a57b7b046861309142da1c5e059df4e437cc87c20` |
| `catalogue/gate_results_e21_2026-08-25.csv` | 1151 | `921d2dfaea3f663562f3f76e52bcd91f603c019c8e9e13ca279bb21ae12bd005` |
| `catalogue/gate_results_e23_2026-08-25.csv` | 2612 | `2ba1283793a9df938e1313f7ddb0e85a9f20b2807c506ea871ff820ef1b13510` |
| `catalogue/gate_results_w1_dev_2026-08-25.csv` | 818 | `7b35d6ddae32dc7701e1ad3b5809aca4d577a1fa15fb58cd49e049d889017987` |
| `catalogue/gate_results_w1_testA_2026-08-25.csv` | 1629 | `20e1fc5941579dcd4871573908a4a51965eadb50b43c0c176bfe2a28ba711cfc` |
| `catalogue/normalization_params_v6_1.json` | 14068 | `b55421f74665c57535fd4e7e82d2dc0633e4c96c426e049a35c6ff2a84fbb827` |
| `catalogue/pixel_disjointness_test_A_2025_2026-08-25.csv` | 3481 | `ce2e9c1baf0e2aa465a112711adbd5568ee9038be6bbb52ba4363a15f109f64c` |
| `catalogue/pixel_disjointness_test_A_2025_2026-08-25_drops.json` | 1851 | `df645cbe9d517966a165a5c1859cf160b6e71eb67074b59bca5fdb987a2c4a25` |
| `catalogue/split_v6_2026-08-25.csv` | 8307 | `5cbbccc44d62bb1d92ec15fd590157da9d4c03762b3ee2eb5e51f15cd854e911` |
| `CITATION.cff` | 4507 | `4dbaf047c44998ac0275cd98aca838bc2e592e5c48a2c4f4c3261c07d054f2dd` |
| `docs/claims_to_evidence.md` | 7889 | `74d1cb7575661a81e75e47b4d0e322e938da21db2496c28cb4cc8adfc8da1a41` |
| `docs/data_access.md` | 4770 | `e1754b200690b493886ecfdabe0883d39971c9c327bf5fa246f74b50a684049c` |
| `docs/integrity_hashes.md` | 15943 | `99fb79f4ad5397fe1b7d53f4e55a526f5e552c1749e9ecd9366879408692f565` |
| `docs/reproducibility.md` | 5342 | `52f2162838d8280ec3cb5d1ad7133841f413a55588c95d34dd28444d78c0add0` |
| `figures/fig1_catalogue.pdf` | 43403 | `d8870fc8035eb61a7364fbbd57830daf1ad2c685623ec8b8ace72df8e077813f` |
| `figures/fig2_framework.pdf` | 42949 | `ce9296e3077661968336a0cf6877ebc7e567c206d33a1464beb55dd4da4dda9b` |
| `figures/fig3_paired.pdf` | 66280 | `68abcfbc1cf7407e4d840bb60f9a74af4adf1b77d8aeb77eb3d0ac046a5b67a5` |
| `figures/make_fig1_catalogue.py` | 8223 | `5a2573e9b9f7b804e87e53d34483cfb26e88980378d0fd2af3be3c21e4dada5b` |
| `figures/make_fig2_framework.py` | 8171 | `9f948d8aa2eb7deefcc93a8a93fa156b22c52bf4889e7b10c3f37751b6a8e631` |
| `figures/make_fig3_paired.py` | 13156 | `389d0b6d59512a2e8e916b5cedea885324bc24453b3ed641c1fc7511b9775bfd` |
| `LICENSE` | 2217 | `293790e15a80eee30d2fceb44ded3e6a7607b4f202cb31c6128c9cbf8eb1e65d` |
| `models/prt_ensemble_m2/e28_grid_summary.json` | 59079 | `ab88006baecbbbf7f2327d0dae077a17afc345200d37b9009a1fafeb41f99534` |
| `models/prt_ensemble_m2/e29_selection.json` | 2966 | `4ee15f8d3b8ab800d11e82919dcb27b464c5c528966b9708301582f965657948` |
| `models/prt_ensemble_m2/m2_ensemble_val_results.json` | 2229 | `9de83b5e06691566bf4531a25c5c6f1dce17a7138a1cc77ce8726ad5dc9e07b6` |
| `models/prt_ensemble_m2/m2_freeze_hashes.json` | 1908 | `27b947348ff4ee493bc46c88a659ad89d1372bfcb9c572cf5ec98b222bf43cfa` |
| `models/prt_ensemble_m2/seed_0.pt` | 1068909 | `7097b786c38e73a634312975cb750e86c092d73dd60dc319a4dbb6aa3c479ca5` |
| `models/prt_ensemble_m2/seed_0_training_record.json` | 4582 | `fcff7f6e2077f3edbd3981d9cc5b56027d322f1a70efdb4448cee08515b2ace3` |
| `models/prt_ensemble_m2/seed_1.pt` | 1068909 | `54b9979bb31463ba3bd75ff7d2ea3bb0fdb895d9fc4014622494ce78efbfc459` |
| `models/prt_ensemble_m2/seed_1_training_record.json` | 4878 | `1c0b46b4380a41af933489ecde718bbda794911ad316af370cffb172ca70ebe8` |
| `models/prt_ensemble_m2/seed_2.pt` | 1068909 | `c32bf3a9a8082a0c832f62898cc49494019848eae9f42c96a8b9a4f2be2e192b` |
| `models/prt_ensemble_m2/seed_2_training_record.json` | 4580 | `d6c2cc495aefeb91af53e0f37a75a9bff70fd0c9d6adf6dbc19b50238705cb77` |
| `models/prt_ensemble_m2/seed_3.pt` | 1068909 | `e8487cffd25d8f3ebc627ccc79f09368691d05223d889774a4746d8736e74cd5` |
| `models/prt_ensemble_m2/seed_3_training_record.json` | 4420 | `b5f0ea6aad6d81e6a693bd397fd7946268f4a30b4bc90f019b5af6342dbedd63` |
| `models/prt_ensemble_m2/seed_4.pt` | 1068909 | `246ad3ecdace5d69eea62d8fe16ea2fc319c798b1b1383c81ae82621bdd9d02f` |
| `models/prt_ensemble_m2/seed_4_training_record.json` | 4574 | `fc96ea50ab27bc57e27693a12fa49d918fa4cf39d72b89f5f91e3f5e72cc4c27` |
| `models/prt_ensemble_m2/seed_5.pt` | 1068909 | `2def69e015e3dc8180d94e1e06df7817b613a6c64f4ca461ec87fe1db62dac30` |
| `models/prt_ensemble_m2/seed_5_training_record.json` | 4738 | `4ee7b9b4b2a21c3c0db34fd56cfdfddebbac0443ab98450a4258fff0c830bc00` |
| `models/prt_ensemble_m2/seed_6.pt` | 1068909 | `ce22771c70b8b9b0c47aeb3e1fc349f2c99a2eea3635f64198712a98a29ab04b` |
| `models/prt_ensemble_m2/seed_6_training_record.json` | 4252 | `ea2ba15364843d290c1959fd378c9220fa261b9f59bcd1e45d76cd747c30f15f` |
| `models/prt_ensemble_m2/seed_7.pt` | 1068909 | `ed589c8abe802ce6f5db96284ce9e7e989c88062a7dcf25a6503688ae03a94ea` |
| `models/prt_ensemble_m2/seed_7_training_record.json` | 4087 | `8e774fcee021536f24a6f0ae1a65896e54b92343fdc1384b2be17487a353423c` |
| `models/prt_ensemble_m2/seed_8.pt` | 1068909 | `f2e2647d020344d9212cd7bf624d814f238cf770f3ab9c5033bd0a1ce7dd7e40` |
| `models/prt_ensemble_m2/seed_8_training_record.json` | 4419 | `23f159def2af60f99de0b5f534cd4df7736ad544a2e2081d308e4d57ab952c07` |
| `models/prt_ensemble_m2/seed_9.pt` | 1068909 | `9cb79735814a097829bd8ba5cbe844c7e64435a470342aa1e58263c1a6536235` |
| `models/prt_ensemble_m2/seed_9_training_record.json` | 4097 | `02cdafc3606f94cfd81361b26037dbd83ba0cd208e4ae76607f2417006509214` |
| `README.md` | 8381 | `6b96760f1acae342729e8da1565530880369ca026b2e30685a4e607bdd21de76` |
| `registration/access_environment_lock.txt` | 342 | `3c602982003cc19e55f1d7d8937e304068ad25b7c7627fe7dba7d1ac5ba501e5` |
| `registration/blockA_input_manifest.csv` | 11418 | `7163744d71085a1bdb3178b162d135265dfc316b367c1894d88f7e616e0122f0` |
| `registration/DEC4_RATIFICATION_ANNEX_2026-08-28.md` | 3544 | `6db0e56b6563dc2596b347efcc479769b1b5fe9bf07a73f2ed84c13c777d54f7` |
| `registration/lock_bundle_prereg_v1.zip` | 57689 | `06beb77fb131d2cdcccafb0bb5ce6800532ba5e7be9c88ffd16349607cb6f71c` |
| `registration/MEMO_IA-1_2026-08-28.md` | 24539 | `eca9f3c0edcce71a90b167b44055c9398a04214099dddfec596fb8b201e91c3a` |
| `registration/PREREG_AMENDMENT_9_2026-08-28.md` | 5069 | `241636d8fe1002f274dcceffd0e6975c56e831a5377b288d80079abd102ae00f` |
| `registration/PREREG_LOCK_PACKAGE_2026-08-26.md` | 16540 | `32e80756f4523f656ccc8006f98dbc27290e144444975b257e0e5ccfa756f8a1` |
| `registration/prereg_test_blocks_v1_LOCKED.md` | 37449 | `9fc24cc719df69643d5a73f0881349be9646e04ff5c4339f6573ae8d88de7c5e` |
| `registration/REGISTRATION_RECORD.md` | 6030 | `6cae7571726a85f13e4acdd22a77d0c46a46b986831f9a3166c0f39f9f853804` |
| `registration/test_event_list_blockA_LOCKED.csv` | 1674 | `b339f1795aaf977ebef0a2e8110d5e4df3c98e0cea0f902f6ec22a4f658b05e3` |
| `requirements.txt` | 725 | `1e581e924fbd65f08b4dfebe4e0d1ed6c45a13db3fe8a8f9dc0bec4cdccea5d3` |
| `results/access_2025/access_20260828T123059Z_per_event.csv` | 31492 | `2c1dd44498b3fe76be7005b493eb0d2424c8347d601efcca1a505418fceb5e79` |
| `results/access_2025/access_20260828T123059Z_results.json` | 1519 | `9f3c1d8410ccb79f2e1841ed2e04f191152d818f8e216f8dc404e7e3b8262226` |
| `results/access_2025/access_20260828T123059Z_strata.json` | 1920 | `414634555287b46d1ccbd862cffd5d108d70376b4f19ca9c34a625229fb435a0` |
| `results/access_2025/access_console_20260828.log` | 3825 | `802ba2ccbab8765a26d2ad6f70de4547b4c583c6f2f3b727021f8853ba574eee` |
| `results/access_2025/access_record_20260828T123059Z.json` | 236 | `9f1d6bf00eac0a71e95d781b34ed6e3b8423a4fc8b55dca3eb20e3bc5cb9b363` |
| `results/encoder_study/decisive_day_by_arch.md` | 4467 | `4978a5a6cb89800fd4d10ce7204707f84acb5124b13d5d755d62b41747eb2ef9` |
| `results/encoder_study/FINDINGS_arch_2026-09-09.md` | 12648 | `8a263d89ed7a4d1f17284aad008ac89cafbc9a322591b4c7449ed56cf3a2f38c` |
| `results/encoder_study/identity_check.json` | 247 | `9ff697bbbf3e774b0a12d3b8fdaa499f7063722a980489730e5ecc6417e9af32` |
| `results/encoder_study/master_results.json` | 68832 | `3e81e66b7081b7c8add0db9b3ed0a5b2d2035455cd4c7d317f9fb51edc3f7757` |
| `results/encoder_study/master_table.md` | 10080 | `78d256daf7850fb56a43bb3a2b39debde5cae2d4f3d38adc979e74ff7600561e` |
| `results/encoder_study/reference_models_per_event.csv` | 55001 | `30b665d4e423c7695971c1c3e0326b1e55b9eb926d03e8ad35e7e35fd4d669b1` |
| `results/encoder_study/val_table.md` | 3519 | `0df712487debffdcc0c7e73cd6754a27ac95b9ce746656d644ea4e4df737488d` |
| `results/exploratory_2026/blockB_inventory.csv` | 3164 | `3c06b0bd06ac46539ee72dae7c2cde5c00baf59dadfaee4206750ad36f272c9a` |
| `results/exploratory_2026/combined_AB_20260908T140257Z_per_event.csv` | 52336 | `b55f9da8539483009b8cffa974565590b40ab5ae3556ee17078e29ef4106db7b` |
| `results/exploratory_2026/decisive_day_20260908T140546Z.csv` | 17041 | `fbeed52fdc0f8de8b664dc639e27e617f68dd30ca61f23bc80a937baabbad0cb` |
| `results/exploratory_2026/exploratory_blockB_20260908T140106Z_per_event.csv` | 22332 | `0ae9318865a31d90fd0050e7d4eb4c16ad845cd6c1d9cc146b4f974ff71d598b` |
| `results/exploratory_2026/exploratory_blockB_20260908T140106Z_results.json` | 1525 | `ca37173c8dc6dcf46d1614d45351f5def58d652fe09102609b2252b92cc68f27` |
| `results/exploratory_2026/exploratory_blockB_20260908T140106Z_strata.json` | 1922 | `21aaa5f7fc9cb41fe4047b6f3ab92729ea628ce1dfbc9e06f990b99a7a8856a6` |
| `results/exploratory_2026/exploratory_record_20260908T140106Z.json` | 3767 | `d555665de2111142cc5374718f79235ce8eb97d85c122a7c5de34e225e2b525c` |
| `results/exploratory_2026/FINDINGS_blockB_2026-09-08.md` | 6632 | `84b9cd8fed4f14cdc7f5462820607b34e638f2b72e8292637654f4628c816a0b` |
| `results/number_register.json` | 7657 | `12762d8696e4b567488869401f9252d3d226f2db5f92bc993ec61a9819dd0d75` |
| `results/REGISTER_VERIFICATION.md` | 7645 | `bd774249e5c9cb67048906829f9990b7612f87b89a4e5f80b07960736583c370` |
| `scripts/acquisition/02_download_all.py` | 12176 | `ef00f8844b20a18ad35e43e1742e7956ace80df3c48566cf182cf284b8668d5e` |
| `scripts/acquisition/03_build_timeseries.py` | 5441 | `f24112032c51cf57e3b09eefbd14b25f53714b1d125599fb323df7201ddfb484` |
| `scripts/acquisition/04_audit_metrics.py` | 8152 | `2f2a824f4fde80fb2b6a0e0154ec0dc141407df870c3a368b71935482c645aa6` |
| `scripts/acquisition/06_era5_download.py` | 20729 | `8b2089efee8942966edc06921c4b9c24fed3ed59a2f42d078e7af6fc389d2293` |
| `scripts/acquisition/_add_event_worker.py` | 17676 | `780052a13245089f24fa50c0b8b8d8af1efd67872ce8a91f49b471418515db63` |
| `scripts/build_dataset_v6.py` | 17027 | `02928d20bb42095301ae2089d139fd264aee2994ac438be39bc4f8937122e139` |
| `scripts/e210_ensemble.py` | 5115 | `64ae5f8d433a09111680fdeb42b64acdef961089ca2db7fafe64bf96dfbcdfe7` |
| `scripts/e28_train_grid.py` | 12616 | `19e70a5051aa905a86bebb6a399835c437ef0e5959728b314bdb5c4d0f37b832` |
| `scripts/e29_selection.py` | 6323 | `5e4427dc81ec07e19a1a7a05b1fd8d280435c16e1e108271b5547d0e525c968f` |
| `scripts/explore_arch_decisive.py` | 7396 | `d1899ecbe9a78a2bc42a0a1e1c0fe76d2e803fc5d6be085301f1f915bc30e47d` |
| `scripts/explore_arch_ensemble.py` | 10030 | `07f97953a5a1814f6d970917c3acd19afa24f52497290c5dae17baff7516ec0e` |
| `scripts/explore_arch_eval.py` | 18356 | `4151ba62842a2df4496ee705c75afdfd9f11ae018ca4715fd7b30e7f313d5ceb` |
| `scripts/explore_arch_train.py` | 17593 | `39be3c66f41dad77e3a2d77d22397b5a92b1460cda4fd538c79df7bf39e76b89` |
| `scripts/explore_decisive_day.py` | 6854 | `0b5a32027195d44105c0d6e34cad6702bde9c25520da2f5bf5cf0a03c600a39d` |
| `scripts/export_event_catalogue_162.py` | 9741 | `6d43ef4fc0a79ca14380773275a877a8b10f47dccf0b74f23dfe4b6aa1f81deb` |
| `scripts/frp_quality_gate.py` | 6396 | `e02db840cf77031d4ba32a33ed712cd126fc15d1d4d75fcaa94222135d725115` |
| `scripts/stage_repro_tree.py` | 5672 | `fd3cccd647b39e61576ff4ae8dfb71a37c4f196eaa0a04c0c8f8393809907c4b` |
| `scripts/test_access_v1.py` | 35506 | `6b403afc53b41a4f0c5bf75b45954ae39cdd35b06ee9d2ca6feef56ee897b1c4` |
| `scripts/verify_register.py` | 13222 | `3573413c176f953e7be22b3a099376a18f2f9f59427d8439cd27168178e60dc8` |

107 files. Generated 2026-09-11.
