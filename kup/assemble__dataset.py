import os
import sys
import pandas as pd
sys.path.insert(0 , '/home/al2644/research/')

def assemble (dir: str):
    entity_pool = pd.read_pickle(os.path.join(dir, 'entity/entity_pool.pickle'))
    update_table = pd.read_pickle(os.path.join(dir, 'update/alpha/update_table.pickle'))
    article_table = pd.read_pickle(os.path.join(dir, 'article/alpha/article_table.pickle'))
    import pdb; pdb.set_trace()

    df = entity_pool.merge(update_table, on = ['entity_id']).merge(article_table, on = ['entity_id'])

    df.to_pickle(os.path.join(dir, 'alpha_dataset.pickle'))

if __name__=='__main__':
    assemble()

