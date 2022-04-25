import base64
import json
import logging
import time
from abc import ABC
from io import BytesIO
from os import path

import tornado.gen
import tornado.web
import tornado.web
from PIL import Image, ImageDraw, ImageFont

from backend.tools import log
from backend.tools.np_encoder import NpEncoder
from model import OcrHandle

logger = logging.getLogger(log.LOGGER_ROOT_NAME + '.' + __name__)

ocrhandle = OcrHandle()

now_time = time.strftime("%Y-%m-%d", time.localtime(time.time()))
from config import dbnet_max_size


class PreProcess(tornado.web.RequestHandler, ABC):
    def get(self):
        self.set_status(404)
        self.write("404 : Please use POST")

    def post(self):
        self.do_det = True
        self.res = []
        short_size = 960
        img_up = self.request.files.get('file', None)
        img_b64 = self.get_argument('img', None)
        compress_size = self.get_argument('compress', None)

        # 判断是上传的图片还是base64
        self.set_header('content-type', 'application/json')
        if img_up is not None and len(img_up) > 0:
            img_up = img_up[0]
            img = Image.open(BytesIO(img_up.body))
        elif img_b64 is not None:
            raw_image = base64.b64decode(img_b64.encode('utf8'))
            img = Image.open(BytesIO(raw_image))
        else:
            self.set_status(400)
            self.res.append({'code': 400, 'msg': '没有传入参数'})
            self.do_det = False
            return

        try:
            if hasattr(img, '_getexif') and img._getexif() is not None:
                orientation = 274
                exif = dict(img._getexif().items())
                if orientation not in exif:
                    exif[orientation] = 0
                if exif[orientation] == 3:
                    img = img.rotate(180, expand=True)
                elif exif[orientation] == 6:
                    img = img.rotate(270, expand=True)
                elif exif[orientation] == 8:
                    img = img.rotate(90, expand=True)
        except Exception as ex:
            self.res.append({'code': 400, 'msg': '产生了一点错误，请检查日志', 'err': str(ex)}, )
            self.do_det = False
            return

        img = img.convert("RGB")
        '''
        是否开启图片压缩
        默认为 960px
        值为 0 时表示不开启压缩
        非 0 时则压缩到该值的大小
        '''
        if compress_size is not None:
            try:
                compress_size = int(compress_size)
            except ValueError:
                self.res.append("短边尺寸参数类型有误，只能是int类型")
                self.do_det = False
            short_size = compress_size
            if short_size < 64:
                self.res.append("短边尺寸过小，请调整短边尺寸")
                self.do_det = False
            short_size = 32 * (short_size // 32)

        img_w, img_h = img.size
        if max(img_w, img_h) * (short_size * 1.0 / min(img_w, img_h)) > dbnet_max_size:
            self.res.append("图片reize后长边过长，请调整短边尺寸")
            self.do_det = False
        self.img = img
        self.short_size = short_size


class TrRun(PreProcess):
    @tornado.gen.coroutine
    def post(self):
        start_time = time.time()
        super().post()
        if self.do_det:
            self.res = ocrhandle.text_predict(self.img, self.short_size)

            img_detected = self.img.copy()
            img_draw = ImageDraw.Draw(img_detected)
            colors = ['red', 'green', 'blue', "purple"]

            for i, r in enumerate(self.res):
                rect, txt, confidence = r
                x1, y1, x2, y2, x3, y3, x4, y4 = rect.reshape(-1)
                size = max(min(x2 - x1, y3 - y2) // 2, 20)

                myfont = ImageFont.truetype(path.join(path.split((path.abspath(__file__)))[0], "sarasa.ttf"), size=size)
                fillcolor = colors[i % len(colors)]
                img_draw.text((x1, y1 - size), str(i + 1), font=myfont, fill=fillcolor)
                for xy in [(x1, y1, x2, y2), (x2, y2, x3, y3), (x3, y3, x4, y4), (x4, y4, x1, y1)]:
                    img_draw.line(xy=xy, fill=colors[i % len(colors)], width=2)

            output_buffer = BytesIO()
            img_detected.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            img_detected_b64 = base64.b64encode(byte_data).decode('utf8')
        else:
            output_buffer = BytesIO()
            self.img.save(output_buffer, format='JPEG')
            byte_data = output_buffer.getvalue()
            img_detected_b64 = base64.b64encode(byte_data).decode('utf8')

        log_info = {'return': self.res}
        logger.info(json.dumps(log_info, cls=NpEncoder))
        self.finish(json.dumps(
            {
                'code': 200,
                'msg': '成功',
                'data': {
                    'img_detected': 'data:image/jpeg;base64,' + img_detected_b64,
                    'raw_out': self.res,
                    'speed_time': round(time.time() - start_time, 2)
                }
            }, cls=NpEncoder))


class TrGetRes(PreProcess):
    @tornado.gen.coroutine
    def post(self):
        start_time = time.time()
        super().post()
        if self.do_det:
            self.res = ocrhandle.text_predict(self.img, self.short_size, False)
        log_info = {'return': self.res}
        logger.info(json.dumps(log_info, cls=NpEncoder))
        self.finish(json.dumps(
            {
                'code': 200,
                'msg': '成功',
                'data': {
                    'raw_out': self.res,
                    'speed_time': round(time.time() - start_time, 2)
                }
            }, cls=NpEncoder))
