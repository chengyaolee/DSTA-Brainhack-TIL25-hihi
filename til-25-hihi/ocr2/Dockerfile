FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_BASE_DIR=/opt/paddleocr_models \
    DET_MODEL_SUBDIR=det/en/en_PP-OCRv3_det_infer \
    FINETUNED_REC_MODEL_DIR_NAME=my_custom_finetuned_ocrv4_rec \
    REC_MODEL_SUBDIR=rec/en/my_custom_finetuned_ocrv4_rec \
    CLS_MODEL_SUBDIR=cls/en/ch_ppocr_mobile_v2.0_cls_infer \
    LAYOUT_MODEL_SUBDIR=layout/en/picodet_lcnet_x1_0_fgd_layout_infer \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip python3-dev wget git \
    libgl1-mesa-glx libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN mkdir -p ${MODELS_BASE_DIR}/det/en ${MODELS_BASE_DIR}/rec/en ${MODELS_BASE_DIR}/cls/en ${MODELS_BASE_DIR}/layout/en \
    && wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O /tmp/det.tar \
    && tar -xf /tmp/det.tar -C ${MODELS_BASE_DIR}/det/en/ --strip-components=0 \
    && wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -O /tmp/cls.tar \
    && tar -xf /tmp/cls.tar -C ${MODELS_BASE_DIR}/cls/en/ --strip-components=0 \
    && wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar -O /tmp/layout.tar \
    && tar -xf /tmp/layout.tar -C ${MODELS_BASE_DIR}/layout/en/ --strip-components=0 \
    && rm /tmp/*.tar \
    && find ${MODELS_BASE_DIR} -name "model.pdmodel" -exec sh -c 'mv "$1" "${1%/*}/inference.pdmodel"' _ {} \; \
    && find ${MODELS_BASE_DIR} -name "model.pdiparams" -exec sh -c 'mv "$1" "${1%/*}/inference.pdiparams"' _ {} \; \
    && echo "=== FINAL VERIFICATION ===" \
    && find ${MODELS_BASE_DIR} -name "inference.pdmodel" \
    && echo "=== END VERIFICATION ==="

COPY my_ppocrv4_rec_english_finetuned_runone/ ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR}/
COPY custom_char_dict.txt /opt/paddleocr_models/dicts/custom_char_dict.txt

RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /workspace/src/

EXPOSE 5003
CMD ["uvicorn", "src.ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]