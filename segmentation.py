import os
import py_vncorenlp

VN_CORE_NLP_DIR = "vncorenlp"


#run this if your machine does not have vncorenlp jar file yet
def download_vn_core_nlp_model(dir):
    py_vncorenlp.download_model(save_dir=dir)


class VnCoreNLPHelper:
    def __init__(self, annotators=["wseg"], max_heap="-Xmx2g", abs_path = os.path.abspath(VN_CORE_NLP_DIR)):
        self.abs_path_of_dir = abs_path
        self.max_heap = max_heap
        self.annotators = annotators
        self.model = self.load_vn_core_nlp()

    def load_vn_core_nlp(self):
        return py_vncorenlp.VnCoreNLP(max_heap_size=self.max_heap, annotators=self.annotators, save_dir=self.abs_path_of_dir)


if __name__ == "__main__":
    abs_path = os.path.abspath(VN_CORE_NLP_DIR)
    core_nlp = VnCoreNLPHelper()
    text = "Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."
    output = core_nlp.model.word_segment(text)
    print(output)

