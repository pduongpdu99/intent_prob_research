# from engine.extraction import ext_from_prompt

# prompt = """
# Chào bạn, công ty chúng tôi hiện tại đang lên kế hoạch phát triển một dự án phần mềm lớn nhằm cung cấp giải pháp tối ưu cho thị trường doanh nghiệp. Chúng tôi muốn xây dựng một ứng dụng tích hợp nhằm quản lý nhân sự và số hóa toàn bộ quy trình làm việc nội bộ. Hệ thống mới này sẽ giúp chuẩn hóa mọi tài liệu, bảo mật thông tin và xử lý dữ liệu tập trung, từ đó cắt giảm đáng kể chi phí vận hành. Về mặt kỹ thuật, nền tảng sử dụng công nghệ đám mây tiên tiến, đòi hỏi một hạ tầng mạng vững chắc để đảm bảo hiệu năng cao. Các chức năng cốt lõi của phần mềm bao gồm tự động hóa chấm công, phê duyệt trực tuyến và xuất báo cáo doanh thu theo thời gian thực. Ngoài ra, các tính năng mở rộng sẽ được tùy biến linh hoạt để đáp ứng đúng yêu cầu khắt khe từ phía khách hàng và các đối tác chiến lược. Khi nhận được sự hỗ trợ phối hợp từ đơn vị bạn, chúng tôi tin rằng việc triển khai cài đặt cho từng đơn vị thành viên sẽ diễn ra đúng tiến độ, mang lại một công cụ làm việc chuyên nghiệp và đồng bộ.
# """

# print(ext_from_prompt(input_prompt=prompt))
from engine.Util import join
from tools.embedding import sentence_embedding, get_similarity_sentence
import pandas
import torch
import gc
from numba import jit, cuda


def build_prompt(title, overview):
    result = "tôi đang cần build một hệ thống \"{0}\" thông tin chi tiết như sau \"{1}\"".format(title, overview)
    return result



@jit(['cuda'])
def runtime():
    dataset = pandas.read_csv(join("data", "_temp.csv")).to_numpy()
    prompts = []
    for title, overview in dataset:
        prompts.append(build_prompt(title, overview))

    sentence = "làm hệ thống quản lý thông tin"
    embeddings = sentence_embedding(prompts)
    sembed = sentence_embedding([sentence])

    xxx = get_similarity_sentence(
        sentence,
        prompts
    )

    print(xxx)

if __name__ == "__main__":
    runtime()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()