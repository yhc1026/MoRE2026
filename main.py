# import sys
# import json
# import os
# import time
# from datetime import datetime
# import math
# import sys
# import hydra
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import colorama
# from colorama import Back, Fore, Style
# from loguru import logger
# from omegaconf import DictConfig, OmegaConf
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# from pathlib import Path
# #import winsound
#
# from utils.core_utils import (
#     get_collator,
#     get_dataset,
#     load_model,
#     set_seed,
#     set_worker_seed,
#     get_optimizer,
#     get_scheduler,
#     BinaryClassificationMetric,
#     TernaryClassificationMetric,
#     EarlyStopping
# )
#
#
# log_path = Path(f'src/log/{datetime.now().strftime("%m%d-%H%M%S")}')
#
#
# class Trainer():
#     def __init__(self,
#                  cfg: DictConfig):
#         self.cfg = cfg
#
#         self.device = 'cuda'
#         # self.task = cfg.task
#         self.task = cfg.data.task
#         if cfg.data.task == 'binary':
#             self.evaluator = BinaryClassificationMetric(self.device)
#         elif cfg.data.task == 'ternary':
#             self.evaluator = TernaryClassificationMetric(self.device)
#         else:
#             raise ValueError('task not supported')
#         self.type = cfg.type
#         self.model_name = cfg.model
#         self.dataset_name = cfg.dataset
#         self.batch_size = cfg.batch_size
#         self.num_epoch = cfg.num_epoch
#         self.generator = torch.Generator().manual_seed(cfg.seed)
#         self.save_path = log_path
#
#         if cfg.type == 'default':
#             self.dataset_range = ['default']
#         else:
#             raise ValueError('experiment type not supported')
#
#         self.collator = get_collator(cfg.model, cfg.dataset, **cfg.data)
#
#     def _reset(self, cfg, fold, type):
#         train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='train', **cfg.data)
#         if hasattr(cfg, 'general') and cfg.general:
#             logger.info(f"Using {cfg.general.dataset} as test dataset!")
#             test_dataset = get_dataset(cfg.model, cfg.general.dataset, fold=fold, split='test', **cfg.data)
#         else:
#             test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
#         if cfg.data.task == 'binary':
#             valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='valid', **cfg.data)
#         # self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#         # self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#         self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,num_workers=0, shuffle=True,generator=self.generator,worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#         self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=0, shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#
#         if cfg.data.task == 'binary':
#            # self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#             self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,num_workers=0, shuffle=False,generator=self.generator,worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#
#         steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
#         self.model = load_model(cfg.model, **dict(cfg.para))
#         self.model.to(self.device)
#         self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
#         self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, **dict(cfg.sche))
#         self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path/'best_model.pth')
#
#     def run(self):
#         acc_list, f1_list, prec_list, rec_list = [], [], [], []
#         a_f1_list, a_prec_list, a_rec_list = [], [], []
#         b_f1_list, b_prec_list, b_rec_list = [], [], []
#         c_f1_list, c_prec_list, c_rec_list = [], [], []
#         for fold in self.dataset_range:
#             self._reset(self.cfg, fold, self.type)
#             logger.info(f'Current fold: {fold}')
#             for epoch in range(self.num_epoch):
#                 logger.info(f'Current Epoch: {epoch}')
#                 self._train(epoch=epoch)
#                 if self.task == 'binary':
#                     self._valid(split='valid', epoch=epoch, use_earlystop=True)
#                     if self.earlystopping.early_stop:
#                         logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
#                         break
#                     self._valid(split='test', epoch=epoch)
#                 elif self.task == 'ternary':
#                     self._valid(split='test', epoch=epoch, use_earlystop=True)
#                     if self.earlystopping.early_stop:
#                         logger.info(f"{Fore.RED}Early stopping at epoch {epoch}")
#                         break
#             logger.info(f'{Fore.RED}Best of Acc in fold {fold}:')
#             self.model.load_state_dict(torch.load(self.save_path/'best_model.pth', weights_only=False))
#             best_metrics = self._valid(split='test', epoch=epoch, final=True)
#             acc_list.append(best_metrics['acc'])
#             f1_list.append(best_metrics['macro_f1'])
#             prec_list.append(best_metrics['macro_prec'])
#             rec_list.append(best_metrics['macro_rec'])
#             a_f1_list.append(best_metrics['a_f1'])
#             a_prec_list.append(best_metrics['a_prec'])
#             a_rec_list.append(best_metrics['a_rec'])
#             b_f1_list.append(best_metrics['b_f1'])
#             b_prec_list.append(best_metrics['b_prec'])
#             b_rec_list.append(best_metrics['b_rec'])
#             if self.task == 'ternary':
#                 c_f1_list.append(best_metrics['c_f1'])
#                 c_prec_list.append(best_metrics['c_prec'])
#                 c_rec_list.append(best_metrics['c_rec'])
#
#         logger.info(f'Best of Acc in all fold: {np.mean(acc_list)}, Best F1: {np.mean(f1_list)}, Best Precision: {np.mean(prec_list)}, Best Recall: {np.mean(rec_list)}')
#         logger.info(f'Best of A F1 in all fold: {np.mean(a_f1_list)}, Best A Precision: {np.mean(a_prec_list)}, Best A Recall: {np.mean(a_rec_list)}')
#         logger.info(f'Best of B F1 in all fold: {np.mean(b_f1_list)}, Best B Precision: {np.mean(b_prec_list)}, Best B Recall: {np.mean(b_rec_list)}')
#         if self.task == 'ternary':
#             logger.info(f'Best of C F1 in all fold: {np.mean(c_f1_list)}, Best C Precision: {np.mean(c_prec_list)}, Best C Recall: {np.mean(c_rec_list)}')
#         #winsound.Beep(500, 1000)
#
#     def _train(self, epoch: int):
#         loss_list =  []
#         loss_pre_list = []
#         self.model.train()
#         pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
#         for batch in pbar:
#             _ = batch.pop('vids')
#             inputs = {key: value.to(self.device) for key, value in batch.items()}
#             labels = inputs.pop('labels')
#
#             output = self.model(**inputs)
#             pred = output['pred'] if isinstance(output, dict) else output
#
#             match self.model.name:
#                 case 'MoRE':
#                     loss, loss_pred = self.model.calculate_loss(**output, label=labels, epoch=epoch)
#                 case _:
#                     loss = F.cross_entropy(pred, labels)
#                     loss_pred = loss
#
#             _, preds = torch.max(pred, 1)
#             self.evaluator.update(preds, labels)
#             loss_list.append(loss.item())
#             loss_pre_list.append(loss_pred.item())
#
#             loss.backward()
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#             self.scheduler.step()
#         metrics = self.evaluator.compute()
#         # print
#         logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")
#
#         logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.5f}, Macro F1: {metrics["macro_f1"]:.5f}, Macro Prec: {metrics["macro_prec"]:.5f}, Macro Rec: {metrics["macro_rec"]:.5f}')
#         logger.info(f'{Fore.BLUE}Train: A F1: {metrics["a_f1"]:.5f}, A Prec: {metrics["a_prec"]:.5f}, A Rec: {metrics["a_rec"]:.5f}')
#         logger.info(f'{Fore.BLUE}Train: B F1: {metrics["b_f1"]:.5f}, B Prec: {metrics["b_prec"]:.5f}, B Rec: {metrics["b_rec"]:.5f}')
#         if self.task == 'ternary':
#             logger.info(f'{Fore.BLUE}Train: C F1: {metrics["c_f1"]:.5f}, C Prec: {metrics["c_prec"]:.5f}, C Rec: {metrics["c_rec"]:.5f}')
#
#     def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
#         loss_list = []
#         self.model.eval()
#         if split == 'valid' and final:
#             raise ValueError('print_wrong only support test split')
#         if split == 'valid':
#             dataloader = self.valid_dataloader
#             split_name = 'Valid'
#             fcolor = Fore.YELLOW
#         elif split == 'test':
#             dataloader = self.test_dataloader
#             split_name = 'Test'
#             fcolor = Fore.RED
#         else:
#             raise ValueError('split not supported')
#         for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
#             vids = batch.pop('vids')
#             inputs = {key: value.to(self.device) for key, value in batch.items()}
#             labels = inputs.pop('labels')
#
#             with torch.no_grad():
#                 output = self.model(**inputs)
#                 pred = output['pred'] if isinstance(output, dict) else output
#                 loss = F.cross_entropy(pred, labels)
#
#             _, preds = torch.max(pred, 1)
#
#             self.evaluator.update(preds, labels)
#             loss_list.append(loss.item())
#         metrics = self.evaluator.compute()
#
#         logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
#         logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']:.5f}, Macro F1: {metrics['macro_f1']:.5f}, Macro Prec: {metrics['macro_prec']:.5f}, Macro Rec: {metrics['macro_rec']:.5f}")
#         logger.info(f"{fcolor}{split_name}: A F1: {metrics['a_f1']:.5f}, A Prec: {metrics['a_prec']:.5f}, A Rec: {metrics['a_rec']:.5f}")
#         logger.info(f"{fcolor}{split_name}: B F1: {metrics['b_f1']:.5f}, B Prec: {metrics['b_prec']:.5f}, B Rec: {metrics['b_rec']:.5f}")
#         if self.task == 'ternary':
#             logger.info(f"{fcolor}{split_name}: C F1: {metrics['c_f1']:.5f}, C Prec: {metrics['c_prec']:.5f}, C Rec: {metrics['c_rec']:.5f}")
#         if use_earlystop:
#             if self.task == 'binary':
#                 self.earlystopping(metrics['acc'], self.model)
#             else:
#                 raise ValueError('task not supported')
#         return metrics
#
# # @hydra.main(version_base=None, config_path=r"D:\code\LAB\MoRE2026\src\config\HateMM_MoRE.yaml")
# @hydra.main(version_base=None, config_path="config", config_name="HateMM_MoRE")
# def main(cfg: DictConfig):
#     logger.remove()
#     logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
#     logger.add(sys.stdout, level="INFO")
#     logger.info(OmegaConf.to_yaml(cfg))
#     pd.set_option('future.no_silent_downcasting', True)
#     colorama.init()
#     set_seed(cfg.seed)
#
#     trainer = Trainer(cfg)
#     trainer.run()
#
# if  __name__ == '__main__':
#     main()

# import sys
# import json
# import os
# import time
# from datetime import datetime
# import math
# import sys
# import hydra
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import colorama
# from colorama import Back, Fore, Style
# from loguru import logger
# from omegaconf import DictConfig, OmegaConf
# from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm
# from pathlib import Path
# #import winsound
#
# from utils.core_utils import (
#     get_collator,
#     get_dataset,
#     load_model,
#     set_seed,
#     set_worker_seed,
#     get_optimizer,
#     get_scheduler,
#     BinaryClassificationMetric,
#     TernaryClassificationMetric,
#     EarlyStopping
# )
#
#
# log_path = Path(f'src/log/{datetime.now().strftime("%m%d-%H%M%S")}')
#
#
# class Trainer():
#     def __init__(self,
#                  cfg: DictConfig):
#         self.cfg = cfg
#
#         self.device = 'cuda'
#         # self.task = cfg.task
#         self.task = cfg.data.task
#         if cfg.data.task == 'binary':
#             self.evaluator = BinaryClassificationMetric(self.device)
#         elif cfg.data.task == 'ternary':
#             self.evaluator = TernaryClassificationMetric(self.device)
#         else:
#             raise ValueError('task not supported')
#         self.type = cfg.type
#         self.model_name = cfg.model
#         self.dataset_name = cfg.dataset
#         self.batch_size = cfg.batch_size
#         self.num_epoch = cfg.num_epoch
#         self.generator = torch.Generator().manual_seed(cfg.seed)
#         self.save_path = log_path
#
#         if cfg.type == 'default':
#             self.dataset_range = ['default']
#         else:
#             raise ValueError('experiment type not supported')
#
#         self.collator = get_collator(cfg.model, cfg.dataset, **cfg.data)
#
#     def _reset(self, cfg, fold, type):
#         train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='train', **cfg.data)
#         if hasattr(cfg, 'general') and cfg.general:
#             logger.info(f"Using {cfg.general.dataset} as test dataset!")
#             test_dataset = get_dataset(cfg.model, cfg.general.dataset, fold=fold, split='test', **cfg.data)
#         else:
#             test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
#         if cfg.data.task == 'binary':
#             valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='valid', **cfg.data)
#         # self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#         # self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#         self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,num_workers=0, shuffle=True,generator=self.generator,worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#         self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=0, shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#
#         if cfg.data.task == 'binary':
#            # self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#             self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,num_workers=0, shuffle=False,generator=self.generator,worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
#
#         steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
#         self.model = load_model(cfg.model, **dict(cfg.para))
#         self.model.to(self.device)
#         self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
#         self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, **dict(cfg.sche))
#         self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path/'best_model.pth')
#
#     def run(self):
#         acc_list, f1_list, prec_list, rec_list = [], [], [], []
#         a_f1_list, a_prec_list, a_rec_list = [], [], []
#         b_f1_list, b_prec_list, b_rec_list = [], [], []
#         c_f1_list, c_prec_list, c_rec_list = [], [], []
#         for fold in self.dataset_range:
#             self._reset(self.cfg, fold, self.type)
#             logger.info(f'Current fold: {fold}')
#             for epoch in range(self.num_epoch):
#                 logger.info(f'Current Epoch: {epoch}')
#                 self._train(epoch=epoch)
#                 if self.task == 'binary':
#                     self._valid(split='valid', epoch=epoch, use_earlystop=True)
#                     if self.earlystopping.early_stop:
#                         logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
#                         break
#                     self._valid(split='test', epoch=epoch)
#                 elif self.task == 'ternary':
#                     self._valid(split='test', epoch=epoch, use_earlystop=True)
#                     if self.earlystopping.early_stop:
#                         logger.info(f"{Fore.RED}Early stopping at epoch {epoch}")
#                         break
#             logger.info(f'{Fore.RED}Best of Acc in fold {fold}:')
#             self.model.load_state_dict(torch.load(self.save_path/'best_model.pth', weights_only=False))
#             best_metrics = self._valid(split='test', epoch=epoch, final=True)
#             acc_list.append(best_metrics['acc'])
#             f1_list.append(best_metrics['macro_f1'])
#             prec_list.append(best_metrics['macro_prec'])
#             rec_list.append(best_metrics['macro_rec'])
#             a_f1_list.append(best_metrics['a_f1'])
#             a_prec_list.append(best_metrics['a_prec'])
#             a_rec_list.append(best_metrics['a_rec'])
#             b_f1_list.append(best_metrics['b_f1'])
#             b_prec_list.append(best_metrics['b_prec'])
#             b_rec_list.append(best_metrics['b_rec'])
#             if self.task == 'ternary':
#                 c_f1_list.append(best_metrics['c_f1'])
#                 c_prec_list.append(best_metrics['c_prec'])
#                 c_rec_list.append(best_metrics['c_rec'])
#
#         logger.info(f'Best of Acc in all fold: {np.mean(acc_list)}, Best F1: {np.mean(f1_list)}, Best Precision: {np.mean(prec_list)}, Best Recall: {np.mean(rec_list)}')
#         logger.info(f'Best of A F1 in all fold: {np.mean(a_f1_list)}, Best A Precision: {np.mean(a_prec_list)}, Best A Recall: {np.mean(a_rec_list)}')
#         logger.info(f'Best of B F1 in all fold: {np.mean(b_f1_list)}, Best B Precision: {np.mean(b_prec_list)}, Best B Recall: {np.mean(b_rec_list)}')
#         if self.task == 'ternary':
#             logger.info(f'Best of C F1 in all fold: {np.mean(c_f1_list)}, Best C Precision: {np.mean(c_prec_list)}, Best C Recall: {np.mean(c_rec_list)}')
#         #winsound.Beep(500, 1000)
#
#     def _train(self, epoch: int):
#         loss_list =  []
#         loss_pre_list = []
#         self.model.train()
#         pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
#         for batch in pbar:
#             _ = batch.pop('vids')
#             inputs = {key: value.to(self.device) for key, value in batch.items()}
#             labels = inputs.pop('labels')
#
#             output = self.model(**inputs)
#             pred = output['pred'] if isinstance(output, dict) else output
#
#             match self.model.name:
#                 case 'MoRE':
#                     loss, loss_pred = self.model.calculate_loss(**output, label=labels, epoch=epoch)
#                 case _:
#                     loss = F.cross_entropy(pred, labels)
#                     loss_pred = loss
#
#             _, preds = torch.max(pred, 1)
#             self.evaluator.update(preds, labels)
#             loss_list.append(loss.item())
#             loss_pre_list.append(loss_pred.item())
#
#             loss.backward()
#             self.optimizer.step()
#             self.optimizer.zero_grad()
#             self.scheduler.step()
#         metrics = self.evaluator.compute()
#         # print
#         logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")
#
#         logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.5f}, Macro F1: {metrics["macro_f1"]:.5f}, Macro Prec: {metrics["macro_prec"]:.5f}, Macro Rec: {metrics["macro_rec"]:.5f}')
#         logger.info(f'{Fore.BLUE}Train: A F1: {metrics["a_f1"]:.5f}, A Prec: {metrics["a_prec"]:.5f}, A Rec: {metrics["a_rec"]:.5f}')
#         logger.info(f'{Fore.BLUE}Train: B F1: {metrics["b_f1"]:.5f}, B Prec: {metrics["b_prec"]:.5f}, B Rec: {metrics["b_rec"]:.5f}')
#         if self.task == 'ternary':
#             logger.info(f'{Fore.BLUE}Train: C F1: {metrics["c_f1"]:.5f}, C Prec: {metrics["c_prec"]:.5f}, C Rec: {metrics["c_rec"]:.5f}')
#
#     def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
#         loss_list = []
#         self.model.eval()
#         if split == 'valid' and final:
#             raise ValueError('print_wrong only support test split')
#         if split == 'valid':
#             dataloader = self.valid_dataloader
#             split_name = 'Valid'
#             fcolor = Fore.YELLOW
#         elif split == 'test':
#             dataloader = self.test_dataloader
#             split_name = 'Test'
#             fcolor = Fore.RED
#         else:
#             raise ValueError('split not supported')
#         for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
#             vids = batch.pop('vids')
#             inputs = {key: value.to(self.device) for key, value in batch.items()}
#             labels = inputs.pop('labels')
#
#             with torch.no_grad():
#                 output = self.model(**inputs)
#                 pred = output['pred'] if isinstance(output, dict) else output
#                 loss = F.cross_entropy(pred, labels)
#
#             _, preds = torch.max(pred, 1)
#
#             self.evaluator.update(preds, labels)
#             loss_list.append(loss.item())
#         metrics = self.evaluator.compute()
#
#         logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
#         logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']:.5f}, Macro F1: {metrics['macro_f1']:.5f}, Macro Prec: {metrics['macro_prec']:.5f}, Macro Rec: {metrics['macro_rec']:.5f}")
#         logger.info(f"{fcolor}{split_name}: A F1: {metrics['a_f1']:.5f}, A Prec: {metrics['a_prec']:.5f}, A Rec: {metrics['a_rec']:.5f}")
#         logger.info(f"{fcolor}{split_name}: B F1: {metrics['b_f1']:.5f}, B Prec: {metrics['b_prec']:.5f}, B Rec: {metrics['b_rec']:.5f}")
#         if self.task == 'ternary':
#             logger.info(f"{fcolor}{split_name}: C F1: {metrics['c_f1']:.5f}, C Prec: {metrics['c_prec']:.5f}, C Rec: {metrics['c_rec']:.5f}")
#         if use_earlystop:
#             if self.task == 'binary':
#                 self.earlystopping(metrics['acc'], self.model)
#             else:
#                 raise ValueError('task not supported')
#         return metrics
#
# # @hydra.main(version_base=None, config_path=r"D:\code\LAB\MoRE2026\src\config\HateMM_MoRE.yaml")
# @hydra.main(version_base=None, config_path="config", config_name="HateMM_MoRE")
# def main(cfg: DictConfig):
#     logger.remove()
#     logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
#     logger.add(sys.stdout, level="INFO")
#     logger.info(OmegaConf.to_yaml(cfg))
#     pd.set_option('future.no_silent_downcasting', True)
#     colorama.init()
#     set_seed(cfg.seed)
#
#     trainer = Trainer(cfg)
#     trainer.run()
#
# if  __name__ == '__main__':
#     main()

import sys
import json
import os
import time
from datetime import datetime
import math
import sys
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import colorama
from colorama import Back, Fore, Style
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
from collections import deque
import copy
# import winsound

from utils.core_utils import (
    get_collator,
    get_dataset,
    load_model,
    set_seed,
    set_worker_seed,
    get_optimizer,
    get_scheduler,
    BinaryClassificationMetric,
    TernaryClassificationMetric,
    EarlyStopping
)

log_path = Path(f'src/log/{datetime.now().strftime("%m%d-%H%M%S")}')


class Trainer():
    def __init__(self,
                 cfg: DictConfig):
        self.cfg = cfg

        self.device = 'cuda'
        # self.task = cfg.task
        self.task = cfg.data.task
        if cfg.data.task == 'binary':
            self.evaluator = BinaryClassificationMetric(self.device)
        elif cfg.data.task == 'ternary':
            self.evaluator = TernaryClassificationMetric(self.device)
        else:
            raise ValueError('task not supported')
        self.type = cfg.type
        self.model_name = cfg.model
        self.dataset_name = cfg.dataset
        self.batch_size = cfg.batch_size
        self.num_epoch = cfg.num_epoch
        self.generator = torch.Generator().manual_seed(cfg.seed)
        self.save_path = log_path

        if cfg.type == 'default':
            self.dataset_range = ['default']
        else:
            raise ValueError('experiment type not supported')

        self.collator = get_collator(cfg.model, cfg.dataset, **cfg.data)

    def _reset(self, cfg, fold, type):
        train_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='train', **cfg.data)
        if hasattr(cfg, 'general') and cfg.general:
            logger.info(f"Using {cfg.general.dataset} as test dataset!")
            test_dataset = get_dataset(cfg.model, cfg.general.dataset, fold=fold, split='test', **cfg.data)
        else:
            test_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='test', **cfg.data)
        if cfg.data.task == 'binary':
            valid_dataset = get_dataset(cfg.model, cfg.dataset, fold=fold, split='valid', **cfg.data)
        # self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=True, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
        # self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
        self.train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
                                           num_workers=0, shuffle=True, generator=self.generator,
                                           worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
        self.test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
                                          num_workers=0, shuffle=False, generator=self.generator,
                                          worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))

        if cfg.data.task == 'binary':
            # self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator, num_workers=min(32, cfg.batch_size//2), shuffle=False, generator=self.generator, worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))
            self.valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, collate_fn=self.collator,
                                               num_workers=0, shuffle=False, generator=self.generator,
                                               worker_init_fn=lambda worker_id: set_worker_seed(worker_id, cfg.seed))

        steps_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)
        self.model = load_model(cfg.model, **dict(cfg.para))
        self.model.to(self.device)
        self.optimizer = get_optimizer(self.model, **dict(cfg.opt))
        self.scheduler = get_scheduler(self.optimizer, steps_per_epoch=steps_per_epoch, **dict(cfg.sche))
        self.earlystopping = EarlyStopping(patience=cfg.patience, path=self.save_path / 'best_model.pth')

    def run(self):
        acc_list, f1_list, prec_list, rec_list = [], [], [], []
        a_f1_list, a_prec_list, a_rec_list = [], [], []
        b_f1_list, b_prec_list, b_rec_list = [], [], []
        c_f1_list, c_prec_list, c_rec_list = [], [], []
        for fold in self.dataset_range:
            self._reset(self.cfg, fold, self.type)
            logger.info(f'Current fold: {fold}')
            for epoch in range(self.num_epoch):
                logger.info(f'Current Epoch: {epoch}')
                self._train(epoch=epoch)
                if self.task == 'binary':
                    self._valid(split='valid', epoch=epoch, use_earlystop=True)
                    if self.earlystopping.early_stop:
                        logger.info(f"{Fore.GREEN}Early stopping at epoch {epoch}")
                        break
                    self._valid(split='test', epoch=epoch)
                elif self.task == 'ternary':
                    self._valid(split='test', epoch=epoch, use_earlystop=True)
                    if self.earlystopping.early_stop:
                        logger.info(f"{Fore.RED}Early stopping at epoch {epoch}")
                        break
            logger.info(f'{Fore.RED}Best of Acc in fold {fold}:')
            self.model.load_state_dict(torch.load(self.save_path / 'best_model.pth', weights_only=False))
            best_metrics = self._valid(split='test', epoch=epoch, final=True)
            acc_list.append(best_metrics['acc'])
            f1_list.append(best_metrics['macro_f1'])
            prec_list.append(best_metrics['macro_prec'])
            rec_list.append(best_metrics['macro_rec'])
            a_f1_list.append(best_metrics['a_f1'])
            a_prec_list.append(best_metrics['a_prec'])
            a_rec_list.append(best_metrics['a_rec'])
            b_f1_list.append(best_metrics['b_f1'])
            b_prec_list.append(best_metrics['b_prec'])
            b_rec_list.append(best_metrics['b_rec'])
            if self.task == 'ternary':
                c_f1_list.append(best_metrics['c_f1'])
                c_prec_list.append(best_metrics['c_prec'])
                c_rec_list.append(best_metrics['c_rec'])

        logger.info(
            f'Best of Acc in all fold: {np.mean(acc_list)}, Best F1: {np.mean(f1_list)}, Best Precision: {np.mean(prec_list)}, Best Recall: {np.mean(rec_list)}')
        logger.info(
            f'Best of A F1 in all fold: {np.mean(a_f1_list)}, Best A Precision: {np.mean(a_prec_list)}, Best A Recall: {np.mean(a_rec_list)}')
        logger.info(
            f'Best of B F1 in all fold: {np.mean(b_f1_list)}, Best B Precision: {np.mean(b_prec_list)}, Best B Recall: {np.mean(b_rec_list)}')
        if self.task == 'ternary':
            logger.info(
                f'Best of C F1 in all fold: {np.mean(c_f1_list)}, Best C Precision: {np.mean(c_prec_list)}, Best C Recall: {np.mean(c_rec_list)}')
        # winsound.Beep(500, 1000)

    def _train(self, epoch: int):
        loss_list = []
        loss_pre_list = []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")
        for batch in pbar:
            _ = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')

            output = self.model(**inputs)
            pred = output['pred'] if isinstance(output, dict) else output

            match self.model.name:
                case 'MoRE':
                    loss, loss_pred = self.model.calculate_loss(**output, label=labels, epoch=epoch)
                case _:
                    loss = F.cross_entropy(pred, labels)
                    loss_pred = loss

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
            loss_pre_list.append(loss_pred.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()
        metrics = self.evaluator.compute()
        # print
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list)}")

        logger.info(
            f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.5f}, Macro F1: {metrics["macro_f1"]:.5f}, Macro Prec: {metrics["macro_prec"]:.5f}, Macro Rec: {metrics["macro_rec"]:.5f}')
        logger.info(
            f'{Fore.BLUE}Train: A F1: {metrics["a_f1"]:.5f}, A Prec: {metrics["a_prec"]:.5f}, A Rec: {metrics["a_rec"]:.5f}')
        logger.info(
            f'{Fore.BLUE}Train: B F1: {metrics["b_f1"]:.5f}, B Prec: {metrics["b_prec"]:.5f}, B Rec: {metrics["b_rec"]:.5f}')
        if self.task == 'ternary':
            logger.info(
                f'{Fore.BLUE}Train: C F1: {metrics["c_f1"]:.5f}, C Prec: {metrics["c_prec"]:.5f}, C Rec: {metrics["c_rec"]:.5f}')

    def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
        loss_list = []
        self.model.eval()
        if split == 'valid' and final:
            raise ValueError('print_wrong only support test split')
        if split == 'valid':
            dataloader = self.valid_dataloader
            split_name = 'Valid'
            fcolor = Fore.YELLOW
        elif split == 'test':
            dataloader = self.test_dataloader
            split_name = 'Test'
            fcolor = Fore.RED
        else:
            raise ValueError('split not supported')
        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')

            with torch.no_grad():
                output = self.model(**inputs)
                pred = output['pred'] if isinstance(output, dict) else output
                loss = F.cross_entropy(pred, labels)

            _, preds = torch.max(pred, 1)

            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
        metrics = self.evaluator.compute()

        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
        logger.info(
            f"{fcolor}{split_name}: Acc: {metrics['acc']:.5f}, Macro F1: {metrics['macro_f1']:.5f}, Macro Prec: {metrics['macro_prec']:.5f}, Macro Rec: {metrics['macro_rec']:.5f}")
        logger.info(
            f"{fcolor}{split_name}: A F1: {metrics['a_f1']:.5f}, A Prec: {metrics['a_prec']:.5f}, A Rec: {metrics['a_rec']:.5f}")
        logger.info(
            f"{fcolor}{split_name}: B F1: {metrics['b_f1']:.5f}, B Prec: {metrics['b_prec']:.5f}, B Rec: {metrics['b_rec']:.5f}")
        if self.task == 'ternary':
            logger.info(
                f"{fcolor}{split_name}: C F1: {metrics['c_f1']:.5f}, C Prec: {metrics['c_prec']:.5f}, C Rec: {metrics['c_rec']:.5f}")
        if use_earlystop:
            if self.task == 'binary':
                self.earlystopping(metrics['acc'], self.model)
            else:
                raise ValueError('task not supported')
        return metrics


class EntropyBasedRetriever:
    """RADAR的熵选择检索器 - 统一维度版本"""

    def __init__(self, memory_size=768, entropy_threshold=0.5, device='cuda'):
        self.memory = deque(maxlen=memory_size)
        self.entropy_threshold = entropy_threshold
        self.device = device

    def compute_entropy(self, logits):
        """计算预测熵"""
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        return entropy

    def _normalize_to_2d(self, tensor):
        """
        将特征标准化为2维 [B, D]
        输入可能是 [B, D], [B, L, D], [L, D], [D] 等
        输出: [B, D]（如果输入是单个样本，则为 [1, D]）
        """
        if tensor is None:
            return None
        t = tensor.clone().detach()

        # 如果是3维 [B, L, D]，平均池化到 [B, D]
        if t.dim() == 3:
            t = t.mean(dim=1)
        # 如果是2维，判断是 [B, D] 还是 [L, D]
        elif t.dim() == 2:
            # 如果第一个维度很小（通常是batch_size，小于100）且第二个维度较大（特征维度，如768），
            # 则认为是 [B, D]，保持不变
            # 如果第一个维度很大（可能是序列长度），则平均池化
            if t.shape[0] > 100:  # 可能是序列长度维度
                t = t.mean(dim=0).unsqueeze(0)  # [L, D] -> [D] -> [1, D]
            # 否则认为是 [B, D]，保持不变
        # 如果是1维 [D]，添加batch维度
        elif t.dim() == 1:
            t = t.unsqueeze(0)  # [D] -> [1, D]

        return t

    def update_memory(self, batch_vids, batch_features, batch_logits, batch_labels=None):
        """更新记忆库 - 存储标准化后的特征"""
        entropies = self.compute_entropy(batch_logits)
        batch_size = batch_logits.shape[0]

        for i in range(batch_size):
            vid = batch_vids[i]

            # 标准化特征到2维 [1, D]，然后取第一个（也是唯一一个）样本
            text_fea = self._normalize_to_2d(batch_features['text_fea'][i:i + 1])
            vision_fea = self._normalize_to_2d(batch_features['vision_fea'][i:i + 1])
            audio_fea = self._normalize_to_2d(batch_features['audio_fea'][i:i + 1])

            # 确保是1维 [D] 用于存储
            if text_fea is not None and text_fea.dim() == 2:
                text_fea = text_fea.squeeze(0)
            if vision_fea is not None and vision_fea.dim() == 2:
                vision_fea = vision_fea.squeeze(0)
            if audio_fea is not None and audio_fea.dim() == 2:
                audio_fea = audio_fea.squeeze(0)

            self.memory.append({
                'vid': vid,
                'text_fea': text_fea.detach().cpu() if text_fea is not None else None,  # [D]
                'vision_fea': vision_fea.detach().cpu() if vision_fea is not None else None,  # [D]
                'audio_fea': audio_fea.detach().cpu() if audio_fea is not None else None,  # [D]
                'logits': batch_logits[i].detach().cpu(),
                'entropy': entropies[i].item(),
                'label': batch_labels[i].item() if batch_labels is not None else None
            })

    def retrieve_stable_references(self, query_features, top_k=5):
        """
        检索稳定的（低熵）参考样本
        返回: 稳定参考样本的字典 [K, D]
        """
        if len(self.memory) < top_k:
            return None

        # 筛选低熵样本
        stable_memory = [m for m in self.memory if m['entropy'] < self.entropy_threshold]
        if len(stable_memory) < top_k:
            stable_memory = list(self.memory)

        # 标准化查询特征到2维 [B, D]
        q_text = self._normalize_to_2d(query_features['text_fea'])
        q_vision = self._normalize_to_2d(query_features['vision_fea'])
        q_audio = self._normalize_to_2d(query_features['audio_fea'])

        # 使用batch中第一个样本进行检索（所有样本共享检索结果）
        q_text_sample = q_text[0:1] if q_text.dim() == 2 else q_text.unsqueeze(0)
        q_vision_sample = q_vision[0:1] if q_vision.dim() == 2 else q_vision.unsqueeze(0)
        q_audio_sample = q_audio[0:1] if q_audio.dim() == 2 else q_audio.unsqueeze(0)

        # 计算与记忆中每个样本的相似度
        similarities = []
        for mem in stable_memory:
            # 记忆中的特征存储在CPU上，需要移到设备上并确保维度正确
            # 记忆特征已经是1维 [D]，需要移到设备上并添加batch维度
            mem_text = mem['text_fea'].to(self.device).unsqueeze(0) if mem['text_fea'] is not None else None
            mem_vision = mem['vision_fea'].to(self.device).unsqueeze(0) if mem['vision_fea'] is not None else None
            mem_audio = mem['audio_fea'].to(self.device).unsqueeze(0) if mem['audio_fea'] is not None else None

            # 计算余弦相似度（都是 [1, D] 形状，都在同一设备上）
            text_sim = F.cosine_similarity(q_text_sample, mem_text, dim=1).item() if mem_text is not None else 0.0
            vision_sim = F.cosine_similarity(q_vision_sample, mem_vision,
                                             dim=1).item() if mem_vision is not None else 0.0
            audio_sim = F.cosine_similarity(q_audio_sample, mem_audio, dim=1).item() if mem_audio is not None else 0.0

            combined_sim = 0.34 * text_sim + 0.33 * vision_sim + 0.33 * audio_sim
            similarities.append(combined_sim)

        # 获取top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # 构建返回结果 - 堆叠成2维 [K, D]
        text_fea_list = []
        vision_fea_list = []
        audio_fea_list = []
        logits_list = []
        entropy_list = []

        for idx in top_indices:
            mem = stable_memory[idx]
            if mem['text_fea'] is not None:
                text_fea_list.append(mem['text_fea'])
            if mem['vision_fea'] is not None:
                vision_fea_list.append(mem['vision_fea'])
            if mem['audio_fea'] is not None:
                audio_fea_list.append(mem['audio_fea'])
            logits_list.append(mem['logits'])
            entropy_list.append(mem['entropy'])

        retrieved = {
            'text_fea': torch.stack(text_fea_list).to(self.device) if text_fea_list else None,  # [K, D]
            'vision_fea': torch.stack(vision_fea_list).to(self.device) if vision_fea_list else None,  # [K, D]
            'audio_fea': torch.stack(audio_fea_list).to(self.device) if audio_fea_list else None,  # [K, D]
            'entropy': torch.tensor(entropy_list).to(self.device),  # [K]
            'logits': torch.stack(logits_list).to(self.device),  # [K, num_classes]
            'similarity': torch.tensor([similarities[i] for i in top_indices]).to(self.device)  # [K]
        }

        return retrieved


class StableAnchorAlignment:
    """稳定锚点对齐模块 - 统一维度版本"""

    def __init__(self, align_loss_weight):
        self.align_loss_weight = align_loss_weight

    def _normalize_to_2d(self, tensor):
        """
        将特征标准化为2维 [B, D] 或 [K, D]
        输入可能是 [B, D], [B, L, D], [K, D], [K, L, D], [L, D], [D] 等
        输出: [B, D] 或 [K, D]
        """
        if tensor is None:
            return None
        t = tensor.clone()

        # 如果是3维 [B, L, D] 或 [K, L, D]，平均池化到 [B, D] 或 [K, D]
        if t.dim() == 3:
            t = t.mean(dim=1)
        # 如果是2维，判断是 [B, D]/[K, D] 还是 [L, D]
        elif t.dim() == 2:
            # 如果第一个维度很大（可能是序列长度），则平均池化
            if t.shape[0] > 100:  # 可能是序列长度维度
                t = t.mean(dim=0).unsqueeze(0)  # [L, D] -> [D] -> [1, D]
            # 否则认为是 [B, D] 或 [K, D]，保持不变
        # 如果是1维 [D]，添加batch维度
        elif t.dim() == 1:
            t = t.unsqueeze(0)  # [D] -> [1, D]

        return t

    def compute_prototype(self, target_features, entropy=None):
        """计算加权原型 - target_features: [K, D] -> [D]"""
        if target_features is None:
            return None

        # 确保是2维 [K, D]
        target_features = self._normalize_to_2d(target_features)
        if target_features is None:
            return None

        if entropy is not None and entropy.numel() > 0:
            weights = F.softmax(-entropy, dim=0)  # [K]
        else:
            weights = torch.ones(target_features.shape[0], device=target_features.device) / target_features.shape[0]

        # [K, D] * [K, 1] -> [K, D] -> [D]
        prototype = (weights.unsqueeze(1) * target_features).sum(dim=0)
        return prototype

    def compute_align_loss(self, query_features, retrieved_dict):
        """
        计算对齐损失 - query_features: [B, D], retrieved_dict: [K, D]
        由于检索结果是共享的，我们计算batch平均特征与检索原型的对齐损失
        """
        if retrieved_dict is None:
            device = next(iter(query_features.values())).device
            return torch.tensor(0.0, device=device)

        align_loss = 0.0
        valid_count = 0

        for modality in ['text', 'vision', 'audio']:
            modality_key = f'{modality}_fea'

            # 处理查询特征 - 标准化为 [B, D]
            query_fea = query_features[modality_key]
            if query_fea is None:
                continue
            query_fea = self._normalize_to_2d(query_fea)  # [B, D] 或 [1, D]

            # 如果只有1个样本，无法计算batch平均，跳过或使用单个样本
            if query_fea.shape[0] == 0:
                continue

            # 处理检索特征 - 标准化为 [K, D]
            retrieved_fea = retrieved_dict.get(modality_key)
            if retrieved_fea is None:
                continue
            retrieved_fea = self._normalize_to_2d(retrieved_fea)  # [K, D]

            entropy = retrieved_dict.get('entropy')

            # 计算检索原型 [D]
            prototype = self.compute_prototype(retrieved_fea, entropy)
            if prototype is None:
                continue

            # 计算batch平均特征 [D]
            # 如果query_fea是 [B, D]，计算batch平均
            if query_fea.dim() == 2:
                batch_mean_fea = query_fea.mean(dim=0)  # [B, D] -> [D]
            elif query_fea.dim() == 1:
                batch_mean_fea = query_fea  # [D]
            else:
                continue

            # 确保prototype是1维
            if prototype.dim() > 1:
                prototype = prototype.mean(dim=0)
            elif prototype.dim() == 0:
                continue

            # 计算余弦相似度（batch平均特征 vs 检索原型）
            cos_sim = F.cosine_similarity(
                batch_mean_fea.unsqueeze(0),
                prototype.unsqueeze(0),
                dim=1
            )
            align_loss += (1.0 - cos_sim)
            valid_count += 1

        if valid_count == 0:
            device = next(iter(query_features.values())).device
            return torch.tensor(0.0, device=device)

        return align_loss / valid_count * self.align_loss_weight


class TargetAwareSelfTraining:
    """目标域自适应自训练"""

    def __init__(self, temperature=0.1, alpha=0.5, beta=0.5):
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta

    def generate_pseudo_labels(self, query_logits, retrieved_dict):
        """
        生成伪标签
        query_logits: [B, num_classes]
        retrieved_dict['logits']: [K, num_classes]
        返回: pseudo_labels [B, num_classes], agg_probs [num_classes]
        """
        if retrieved_dict is None or retrieved_dict.get('logits') is None:
            return None, None

        query_probs = F.softmax(query_logits, dim=-1)  # [B, num_classes]
        retrieved_probs = F.softmax(retrieved_dict['logits'], dim=-1)  # [K, num_classes]

        similarities = retrieved_dict.get('similarity')
        if similarities is None or similarities.numel() == 0:
            # 如果没有相似度，使用均匀权重
            weights = torch.ones(retrieved_probs.shape[0], device=retrieved_probs.device) / retrieved_probs.shape[0]
        else:
            weights = F.softmax(similarities / self.temperature, dim=0)  # [K]

        # 加权聚合检索的概率 [K, num_classes] * [K, 1] -> [num_classes]
        agg_probs = (weights.unsqueeze(1) * retrieved_probs).sum(dim=0)  # [num_classes]

        # 扩展到batch维度并组合
        batch_size = query_probs.shape[0]
        agg_probs_expanded = agg_probs.unsqueeze(0).expand(batch_size, -1)  # [B, num_classes]

        pseudo_labels = self.alpha * query_probs + self.beta * agg_probs_expanded  # [B, num_classes]

        return pseudo_labels, agg_probs


class MoRERADARTrainer(Trainer):
    """增强RADAR机制的Trainer - 只在测试时使用"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # RADAR组件
        self.retriever = EntropyBasedRetriever(
            memory_size=cfg.para.get('memory_size', 768),
            entropy_threshold=cfg.para.get('entropy_threshold', 0.2),
            device=self.device
        )

        self.alignment = StableAnchorAlignment(
            align_loss_weight=cfg.para.get('align_loss_weight', 0.5)
        )

        self.self_training = TargetAwareSelfTraining(
            temperature=cfg.para.get('self_train_temperature', 0.2),
            alpha=cfg.para.get('pseudo_alpha', 0.7),
            beta=cfg.para.get('pseudo_beta', 0.3)
        )

        # RADAR配置
        self.use_radar = cfg.para.get('use_radar', True)
        self.radar_at_test = cfg.para.get('radar_at_test', True)  # 只在测试时使用
        self.num_retrieve = cfg.para.get('num_retrieve', 2)

        # 历史batch缓存（用于测试时检索）
        self.history_batches = deque(maxlen=cfg.para.get('history_size', 1000))

    def _extract_features(self, inputs, output):
        """从输入和输出中提取特征"""
        features = {}

        # 原始特征
        features['text_fea'] = inputs.get('text_fea', output.get('text_fea', None))
        features['vision_fea'] = inputs.get('vision_fea', output.get('vision_fea', None))
        features['audio_fea'] = inputs.get('audio_fea', output.get('audio_fea', None))

        # 增强后的特征（如果有）
        if 'text_fea_aug' in output:
            features['text_fea_aug'] = output['text_fea_aug']
        if 'vision_fea_aug' in output:
            features['vision_fea_aug'] = output['vision_fea_aug']
        if 'audio_fea_aug' in output:
            features['audio_fea_aug'] = output['audio_fea_aug']

        return features

    def _train(self, epoch: int):
        """训练时完全不用RADAR，保持原始训练逻辑"""
        loss_list = []
        loss_pre_list = []
        self.model.train()
        pbar = tqdm(self.train_dataloader, bar_format=f"{Fore.BLUE}{{l_bar}}{{bar}}{{r_bar}}")

        for batch in pbar:
            _ = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')

            output = self.model(**inputs)
            pred = output['pred'] if isinstance(output, dict) else output

            match self.model.name:
                case 'MoRE':
                    loss, loss_pred = self.model.calculate_loss(**output, label=labels, epoch=epoch)
                case _:
                    loss = F.cross_entropy(pred, labels)
                    loss_pred = loss

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())
            loss_pre_list.append(loss_pred.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

        metrics = self.evaluator.compute()
        logger.info(f"{Fore.BLUE}Train: Loss: {np.mean(loss_list):.5f}")
        logger.info(f'{Fore.BLUE}Train: Acc: {metrics["acc"]:.5f}, Macro F1: {metrics["macro_f1"]:.5f}, '
                    f'Macro Prec: {metrics["macro_prec"]:.5f}, Macro Rec: {metrics["macro_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Train: A F1: {metrics["a_f1"]:.5f}, A Prec: {metrics["a_prec"]:.5f}, '
                    f'A Rec: {metrics["a_rec"]:.5f}')
        logger.info(f'{Fore.BLUE}Train: B F1: {metrics["b_f1"]:.5f}, B Prec: {metrics["b_prec"]:.5f}, '
                    f'B Rec: {metrics["b_rec"]:.5f}')
        if self.task == 'ternary':
            logger.info(f'{Fore.BLUE}Train: C F1: {metrics["c_f1"]:.5f}, C Prec: {metrics["c_prec"]:.5f}, '
                        f'C Rec: {metrics["c_rec"]:.5f}')

    def _valid(self, split: str, epoch: int, use_earlystop=False, final=False):
        """验证/测试时根据配置决定是否使用RADAR"""
        loss_list = []

        if split == 'valid' and final:
            raise ValueError('print_wrong only support test split')

        if split == 'valid':
            dataloader = self.valid_dataloader
            split_name = 'Valid'
            fcolor = Fore.YELLOW
            use_radar = False  # 验证集不用RADAR
        elif split == 'test':
            dataloader = self.test_dataloader
            split_name = 'Test'
            fcolor = Fore.RED
            use_radar = self.radar_at_test  # 测试时根据配置决定
        else:
            raise ValueError('split not supported')

        self.model.eval()

        for batch in tqdm(dataloader, bar_format=f"{fcolor}{{l_bar}}{{bar}}{{r_bar}}"):
            vids = batch.pop('vids')
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            labels = inputs.pop('labels')

            with torch.no_grad():
                # 基础预测
                output = self.model(**inputs)
                pred = output['pred'] if isinstance(output, dict) else output

                # === 测试时RADAR增强 ===
                if use_radar:
                    # 提取特征
                    features = self._extract_features(inputs, output)

                    # 检索稳定参考样本
                    retrieved = self.retriever.retrieve_stable_references(
                        features,
                        top_k=self.num_retrieve
                    )

                    # 如果有检索结果，用检索结果增强预测
                    if retrieved is not None:
                        # 方法1：直接平均（简单）
                        # retrieved_probs = F.softmax(retrieved['logits'], dim=-1)
                        # enhanced_probs = (F.softmax(pred, dim=-1) + retrieved_probs.mean(dim=0)) / 2
                        # pred = torch.log(enhanced_probs + 1e-10)

                        # 方法2：相似度加权（更精细）
                        similarities = retrieved['similarity']  # [K]
                        weights = F.softmax(similarities / 0.1, dim=0)  # 温度参数

                        retrieved_probs = F.softmax(retrieved['logits'], dim=-1)  # [K, C]
                        agg_retrieved = (weights.unsqueeze(1) * retrieved_probs).sum(dim=0)  # [C]

                        # 融合：当前预测(0.7) + 检索结果(0.3)
                        enhanced_probs = 0.7 * F.softmax(pred, dim=-1) + 0.3 * agg_retrieved.unsqueeze(0)
                        pred = torch.log(enhanced_probs + 1e-10)

                    # 更新记忆库（用于后续检索）
                    self.retriever.update_memory(vids, features, pred, labels)

                loss = F.cross_entropy(pred, labels)

            _, preds = torch.max(pred, 1)
            self.evaluator.update(preds, labels)
            loss_list.append(loss.item())

        metrics = self.evaluator.compute()

        logger.info(f"{fcolor}{split_name}: Loss: {np.mean(loss_list):.5f}")
        logger.info(f"{fcolor}{split_name}: Acc: {metrics['acc']:.5f}, Macro F1: {metrics['macro_f1']:.5f}, "
                    f"Macro Prec: {metrics['macro_prec']:.5f}, Macro Rec: {metrics['macro_rec']:.5f}")
        logger.info(f"{fcolor}{split_name}: A F1: {metrics['a_f1']:.5f}, A Prec: {metrics['a_prec']:.5f}, "
                    f"A Rec: {metrics['a_rec']:.5f}")
        logger.info(f"{fcolor}{split_name}: B F1: {metrics['b_f1']:.5f}, B Prec: {metrics['b_prec']:.5f}, "
                    f"B Rec: {metrics['b_rec']:.5f}")
        if self.task == 'ternary':
            logger.info(f"{fcolor}{split_name}: C F1: {metrics['c_f1']:.5f}, C Prec: {metrics['c_prec']:.5f}, "
                        f"C Rec: {metrics['c_rec']:.5f}")

        if use_earlystop:
            if self.task == 'binary':
                self.earlystopping(metrics['acc'], self.model)
            else:
                raise ValueError('task not supported')

        return metrics


@hydra.main(version_base=None, config_path="config", config_name="HateMM_MoRE")
def main(cfg: DictConfig):
    logger.remove()
    logger.add(log_path / 'log.log', retention="10 days", level="DEBUG")
    logger.add(sys.stdout, level="INFO")
    logger.info(OmegaConf.to_yaml(cfg))
    pd.set_option('future.no_silent_downcasting', True)
    colorama.init()
    set_seed(cfg.seed)

    # 根据配置选择使用普通Trainer还是RADAR增强的Trainer
    if cfg.para.get('use_radar', False):
        logger.info(f"{Fore.GREEN}Using RADAR-enhanced Trainer")
        trainer = MoRERADARTrainer(cfg)
    else:
        logger.info(f"{Fore.GREEN}Using standard Trainer")
        trainer = Trainer(cfg)

    trainer.run()


if __name__ == '__main__':
    main()