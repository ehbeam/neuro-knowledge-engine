# A computational knowledge engine for human neuroscience

Code repository for a manuscript in preparation by Elizabeth Beam, Christopher Potts, Russell Poldrack, & Amit Etkin

## Abstract

Functional neuroimaging has been a mainstay of human neuroscience for the past 25 years. Interpretation of fMRI data has often occurred within knowledge frameworks crafted by experts, which have the potential to reify historical trends and amplify biases that limit the replicability of findings. Here, we employ a computational approach to derive a data-driven framework for neurobiological domains that synthesizes the texts and data of nearly 20,000 human neuroimaging articles. Crucially, the structure-function links in each domain better replicate across held-out articles than those mapped from dominant frameworks in neuroscience and psychiatry. We further show that the data-driven framework partitions the literature into modular subfields, for which the domains serve as generalizable prototypes of structure-function patterns observed in single articles. The approach to computational ontology we present here is the most comprehensive characterization of human brain circuits quantifiable with fMRI, and moreover, can be extended to synthesize other scientific literatures.


## Index of Figures

### Main Text

| Figure   | Files                                                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 1b       | ontology/ontol\_data-driven\_lr.ipynb, ontology/ontology.py                                                                   |
| 1c       | partition/part\_splits.ipynb, partition/partition.py                                                                          |
| 1d       | modularity/mod\_kvals\_lr.ipynb                                                                                               |
| 1e       | prototype/proto\_kvals\_lr.ipynb                                                                                              |
| 2b       | ontology/ontol\_rdoc.ipynb, ontology/ontology.py                                                                              |
| 3a       | ontology/ontol\_rdoc.ipynb, ontol\_sim\_lr.ipynb, ontology/ontology.py                                                        |
| 3b       | ontology/ontol\_data-driven\_lr.ipynb, ontol\_sim\_lr.ipynb, ontology/ontology.py                                             |
| 3c       | ontology/ontol\_ontol_dsm.ipynb, ontol\_sim\_lr.ipynb, ontology/ontology.py                                                   |
| 4b, e    | prediction/pred\_data-driven_lr.ipynb, prediction/logistic\_regression/prediction.py, prediction/evaluation.py                |
| 4c, f    | prediction/pred\_rdoc.ipynb, prediction/logistic\_regression/prediction.py, prediction/evaluation.py                          |
| 4d, g    | prediction/pred\_dsm.ipynb, prediction/logistic\_regression/prediction.py, prediction/evaluation.py                           |
| 4h       | prediction/comp\_frameworks\_lr.ipynb                                                                                         |
| 5a-f     | mds/mds.ipynb, mds/mds.py                                                                                                     |
| 5g       | modularity/mod\_data-driven\_lr.ipynb, modularity/modularity.py                                                               |
| 5h       | modularity/mod\_rdoc.ipynb, modularity/modularity.py                                                                          |
| 5i       | modularity/mod\_dsm.ipynb, modularity/modularity.py                                                                           |
| 5j       | modularity/comp\_frameworks\_lr.ipynb, modularity/modularity.py                                                               |
| 5k       | prototype/proto\_data-driven\_lr.ipynb, prototype/prototype.py                                                                |
| 5l       | prototype/proto\_rdoc.ipynb, prototype/prototype.py                                                                           |
| 5m       | prototype/proto\_dsm.ipynb, prototype/prototype.py                                                                            |
| 5n       | prototype/comp\_frameworks\_lr.ipynb, prototype/prototype.py                                                                  |

### Supplementary Material

| Figure   | Files                                                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 1        | corpus/cohorts.ipynb                                                                                                          |
| 2        | validation/val_brainmap_top.ipynb                                                                                             |
| 3        | validation/val_brainmap_sims.ipynb                                                                                            |
| 4-5      | ontology/ontol\_kvals\_lr.ipynb, ontology/ontology.py                                                                         |
| 6a-b     | ontology/ontol\_data-driven\_nn.ipynb, ontology/ontology.py                                                                   |
| 6c       | mds/mds.ipynb, mds/mds.py                                                                                                     |
| 6d       | modularity/mod\_data-driven\_nn.ipynb, modularity/modularity.py                                                               |
| 6e       | prototype/proto\_data-driven\_nn.ipynb, prototype/prototype.py                                                                |
| 7-8      | ontology/ontol\_kvals\_nn.ipynb, ontology/ontology.py                                                                         |
| 9        | hierarchy/hier_data-driven_lr.ipynb                                                                                           |
| 10       | stability/stab_data-driven_lr_top.ipynb                                                                                       |
| 11a      | ontology/ontol_data-driven_terms.ipynb, ontology/ontol_sim_terms.ipynb, ontology/ontology.py                                  |
| 11b-e    | ontology/ontol_sim_terms.ipynb                                                                                                |
| 12a, d; 13b, e; 14a, d | prediction/pred\_data-driven_lr.ipynb, prediction/logistic\_regression/prediction.py, prediction/evaluation.py  |
| 12b, e; 13c, f; 14b, e | prediction/pred\_rdoc.ipynb, prediction/logistic\_regression/prediction.py, prediction/evaluation.py            |
| 12c, f; 13d, g; 14c, f | prediction/pred\_dsm.ipynb, prediction/logistic\_regression/prediction.py, prediction/evaluation.py             |
| 12g, 13h-j, 14g-i | prediction/comp\_frameworks\_lr.ipynb                                                                                |
| 15b, e; 16a, d; 17b, e; 18a, d | prediction/pred\_data-driven_nn.ipynb, prediction/neural\_network/sherlock/neural\_network.py, prediction/evaluation.py |
| 15c, f; 16b, e; 17c, f; 18b, e | prediction/pred\_rdoc.ipynb, prediction/neural\_network/sherlock/neural\_network.py, prediction/evaluation.py |
| 15d, g; 16c, f; 17d, g; 18c, f | prediction/pred\_dsm.ipynb, prediction/neural\_network/sherlock/neural\_network.py, prediction/evaluation.py |
| 15h, 16g, 17h-j, 18g-i | prediction/comp\_frameworks\_nn.ipynb                                                                           |
| 19a, d   | prediction/comp_frameworks_lr_k09.ipynb                                                                                       |
| 19b-c, e-f | prediction/pred_data-driven_lr_k09.ipynb                                                                                    |
| 19g-h    | partition/part_data-driven_lr_k09.ipynb, mds/mds.ipynb                                                                        |
| 19i Left | modularity/comp_frameworks_lr_k09.ipynb                                                                                       |
| 19i Right | modularity/mod_data-driven_lr_k09.ipynb                                                                                      |
| 19j Left | prototype/comp_frameworks_lr_k09.ipynb                                                                                        |
| 19j Right | prototype/proto_data-driven_lr_k09.ipynb                                                                                     |
| 20a      | partition/part\_data-driven\_lr.ipynb, partition/partition.py                                                                 |
| 20b      | partition/part\_rdoc.ipynb, partition/partition.py                                                                            |
| 20c      | partition/part\_dsm.ipynb, partition/partition.py                                                                             |
| 20d-f    | tsne/tsne.ipynb                                                                                                               |

| Table    | Files                                                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------------------------------------- |
| 1        | data/data_table_coord.ipynb                                                                                                   |
| 2        | lexicon/preproc\_cogneuro.py, lexicon/preproc\_psychiatry.py, lexicon/preproc\_rdoc.py, lexicon/preproc\_dsm.py               |
| 3        | data/text/pubmed/gen\_190428/query.txt, data/text/pubmed/psy\_190428/query.txt                                                |
| 4-5      | prediction/table\_lr-nn.ipynb                                                                                                 |

