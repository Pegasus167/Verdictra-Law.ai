import sys
sys.stdout.reconfigure(encoding='utf-8')
from pipeline.kge_trainer import _train_kge
_train_kge('celir_llp_vs_midc')
print('KGE done')
