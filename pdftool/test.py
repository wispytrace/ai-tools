from pdf2zh import translate, translate_stream
from pdf2zh.doclayout import OnnxModel, ModelInstance
from pdf2zh.doclayout import OnnxModel, ModelInstance

ModelInstance.value = OnnxModel.load_available()
print("Loaded BABELDOC_MODEL:", ModelInstance.value)
params = {
    'lang_in': 'en',
    'lang_out': 'zh',
    'service': 'bing',
    'thread': 4,
    'model': ModelInstance.value
}

(file_mono, file_dual) = translate(files=['/root/bai2009.pdf'], **params)[0]