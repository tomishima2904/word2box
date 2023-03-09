import re
import argparse
from pathlib import Path
import datetime, json

import logzero
from logzero import logger
from gensim.models.word2vec import LineSentence, Word2Vec


logger_word2vec = logzero.setup_logger(name='gensim.models.word2vec')
logger_base_any2vec = logzero.setup_logger(name='gensim.models.base_any2vec')

regex_entity = re.compile(r'##[^#]+?##')


def main(args):
    # Get datetime for name of dir
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    date_time = now.strftime('%y%m%d%H%M%S')

    output_dir = Path(f"{args.output_dir}/{date_time}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dump config as .json
    with open(f"{output_dir}/config.json", 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    all_vectors_file = output_dir / 'all_vectors.txt'

    logger.info('training the model')
    model = Word2Vec(sentences=LineSentence(args.corpus_file),
                     vector_size=args.embed_size,
                     window=args.window_size,
                     negative=args.sample_size,
                     min_count=args.min_count,
                     workers=args.workers,
                     sg=1,
                     hs=0,
                     epochs=args.epochs,
                     seed=19990429)


    total_vocab_size = len(model.wv)
    logger.info(f'total vocabulary size: {total_vocab_size}')

    logger.info('writing word/entity vectors to files')
    with open(all_vectors_file, 'w') as fo_all:

        # write word2vec headers to each file
        print(total_vocab_size, args.embed_size, file=fo_all)

        # write tokens and vectors
        for (token, _) in model.wv.key_to_index.items():
            vector = model.wv.get_vector(token)
            print(token, *vector, file=fo_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_file', type=str, required=True,
        help='Corpus file (.txt)')
    parser.add_argument('--output_dir', type=str, required=True,
        help='Output directory to save embedding files')
    parser.add_argument('--embed_size', type=int, default=100,
        help='Dimensionality of the word/entity vectors [100]')
    parser.add_argument('--window_size', type=int, default=5,
        help='Maximum distance between the current and '
             'predicted word within a sentence [5]')
    parser.add_argument('--sample_size', type=int, default=5,
        help='Number of negative samples [5]')
    parser.add_argument('--min_count', type=int, default=5,
        help='Ignores all words/entities with total frequency lower than this [5]')
    parser.add_argument('--epochs', type=int, default=5,
        help='number of training epochs [5]')
    parser.add_argument('--workers', type=int, default=2,
        help='Use these many worker threads to train the model [2]')
    args = parser.parse_args()
    main(args)
