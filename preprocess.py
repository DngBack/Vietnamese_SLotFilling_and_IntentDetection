import string
import re
from underthesea import text_normalize

def remove_dup(text):

    def replace(match):
        m = match.group(0)
        if d[m[0]] == d[m[1]]:
            return m[0]
        else:
            return m[0] + m[1]

    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
    uniChars += string.ascii_letters
    unsignChars += string.ascii_letters

    d = {k: v for (k, v) in zip(uniChars, unsignChars)}
    return re.sub(fr'\S([{uniChars}])\1+', replace, text)


def normalize_text(text):
    text = text_normalize(text)
    text = remove_dup(text)
    return text