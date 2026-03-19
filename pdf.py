from pypdf import PdfReader, PdfWriter
from copy import deepcopy

def split_pdf(input_path, output_path):
    reader = PdfReader(input_path)
    writer = PdfWriter()
    num_pages = len(reader.pages)

    for i in range(num_pages):
        page = reader.pages[i]
        
        # 첫 번째와 마지막 페이지는 그대로 추가
        if i == 0 or i == num_pages - 1:
            writer.add_page(page)
            continue

        # 원본 좌표 정보 가져오기
        width = page.mediabox.width
        height = page.mediabox.height

        # 1. 왼쪽 페이지 작업
        left_page = deepcopy(page)
        left_page.mediabox.upper_right = (width / 2, height)
        left_page.mediabox.lower_left = (0, 0)
        writer.add_page(left_page)

        # 2. 오른쪽 페이지 작업
        right_page = deepcopy(page)
        right_page.mediabox.upper_right = (width, height)
        right_page.mediabox.lower_left = (width / 2, 0)
        writer.add_page(right_page)

    with open(output_path, "wb") as f:
        writer.write(f)

split_pdf("궁금해궁금해.pdf", "궁금해궁금해_분할.pdf")