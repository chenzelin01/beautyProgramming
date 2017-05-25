import sys
import os
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path += [curdir]
import argparse
from tagging_evaluation_util import get_tagging_results
from evaluation_stats_util import F1Stats
from raw_result_parser import iter_results
from ioutil import open_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_prediction')
    parser.add_argument('test_file')
    parser.add_argument('-s', '--schema', default='BIO2',
            choices=['BO', 'BO2', 'BIO', 'BIO2', 'BIO3'])
    parser.add_argument('-o', '--output', default='-')
    parser.add_argument('-f', '--fuzzy', action='store_true',
                        help='fuzzy evaluation')
    options = parser.parse_args()

    stats = F1Stats(options.fuzzy)
    for q_tokens, e_tokens, tags, golden_answers in \
            iter_results(options.raw_prediction, options.test_file,
                    options.schema):
        if q_tokens is None: continue # one question has been processed
        pred_answers = get_tagging_results(e_tokens, tags)
        stats.update(golden_answers, pred_answers)

    output = sys.stdout if options.output == '-' else open_file(options.output, 'w')
    print >> output, stats.get_metrics_str()
    output.close()


if __name__ == '__main__':
    main()
