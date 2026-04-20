import sys
sys.stdout.reconfigure(encoding='utf-8')

from pipeline.kge_trainer import _fetch_triples, _train_kge

print('Testing _fetch_triples...')
try:
    heads, relations, tails = _fetch_triples('celir_llp_vs_midc')
    print(f'Triples: {len(heads)}')
    if heads:
        print(f'Sample: {heads[0]} --[{relations[0]}]--> {tails[0]}')
    else:
        print('NO TRIPLES FOUND - graph is empty or has no case_id filter match')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()

print()
print('Testing _train_kge...')
try:
    _train_kge('celir_llp_vs_midc')
    print('Training complete')
except Exception as e:
    print(f'ERROR: {e}')
    import traceback; traceback.print_exc()
