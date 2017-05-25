# -*- coding:utf-8 -*-
from langconv import *
import sys
import io
import multiprocessing
import jieba



class MyLineProcess(multiprocessing.Process):
    def __init__(self,  lock, lines, outputfile, group_=None, target_=None, name_=None, args_=(), kwargs_={}):
        multiprocessing.Process.__init__(self, group=group_, target=target_, name=name_, args=args_, kwargs=kwargs_)
        self.outputfile = outputfile
        self.lock = lock
        self.lines = lines

    @staticmethod
    def handle_line(lock, lines, outputfile):
        line = MyLineProcess.traditional_to_simplified(lines)
        line_cut = jieba.cut(line)
        write_res = " ".join(line_cut)
        lock.acquire()
        with io.open(outputfile, mode='a', encoding='utf-8') as outfile:
            outfile.write(write_res)
            outfile.write(u'\n')
        lock.release()
        return write_res

    def run(self):
        # multiprocessing.Process.run(self)
        MyLineProcess.handle_line(self.lock, self.lines, self.outputfile)

    @staticmethod
    def traditional_to_simplified(sentence):
        sentence = Converter('zh-hans').convert(sentence)
        return sentence
    @staticmethod
    def simplified_to_traditional(sentence):
        sentence = Converter('zh-hant').convert(sentence)
        return sentence

def cut_words(inputfile, outputfile):
    # in window platform
    # multiprocessing.freeze_support()
    count = 1
    current_pro = 0
    temp_lines = ""
    process_list = []
    lock = multiprocessing.Lock()
    max_process_num = multiprocessing.cpu_count()
    # process_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    with io.open(inputfile, mode='r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip()
            if len(line) < 1:
                continue
            if line.startswith('<doc'):
                temp_lines = ""
            elif line != '</doc>':
                temp_lines += line

            if line == '</doc>':
                count += 1
                # cache.append(process_pool.apply_async(func=handle_line, args=(lock, temp_lines, outputfile)))
                if len(process_list) < max_process_num - 1:
                    tp = MyLineProcess(lock, temp_lines, outputfile)
                    tp.start()
                else:
                    join_pro = max_process_num - current_pro - 1
                    process_list[join_pro].join()
                    process_list[join_pro].lines = temp_lines
                    process_list[join_pro].start()
                current_pro += 1
                process_list.append(tp)
            if current_pro > max_process_num - 1:
                current_pro = 0
            if count % 1000 == 0:
                print('%s sentences have been processed ' %count)
        # process_pool.close()
        # process_pool.join()
        for p in process_list:
            p.join()
if __name__ == '__main__':
    inputfile = 'wiki_00'
    outputfile = 'wiki_split'
    # this option is only can used in linux environment
    # jieba.enable_parallel(multiprocessing.cpu_count())
    cut_words(inputfile, outputfile)

