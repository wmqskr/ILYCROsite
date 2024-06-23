#删除假代码X
def remove_O_from_file(filename):
    # 读取原始文件并删除所有 'X' 字符
    with open(filename, 'r') as file:
        content = file.read()
        content_without_O = content.replace('O', '')
        content_without_U = content_without_O.replace('U', '')
    # 将处理后的数据写入临时文件
    with open('./dataset/test/Kcr_INDP_cleaned.txt', 'w') as cleaned_file:
        cleaned_file.write(content_without_U)

# 调用函数
# remove_O_from_file('./dataset/train/kcr_cvN.txt')
# remove_O_from_file('./dataset/train/kcr_cvP.txt')
# remove_O_from_file('./dataset/test/Kcr_INDN.txt')
remove_O_from_file('./dataset/test/Kcr_INDP.txt')