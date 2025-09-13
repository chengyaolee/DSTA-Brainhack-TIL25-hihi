FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04



# Note: Update FINETUNED_REC_MODEL_DIR_NAME and REC_MODEL_SUBDIR if your new model
# directory has a different name.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive \
    MODELS_BASE_DIR=/opt/paddleocr_models \
    DET_MODEL_SUBDIR=det/en/en_PP-OCRv3_det_infer \
    FINETUNED_REC_MODEL_DIR_NAME=my_new_custom_model_default_dict \
    REC_MODEL_SUBDIR=rec/en/my_new_custom_model_default_dict \
    CLS_MODEL_SUBDIR=cls/en/ch_ppocr_mobile_v2.0_cls_infer \
    LAYOUT_MODEL_SUBDIR=layout/en/picodet_lcnet_x1_0_fgd_layout_infer \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=/workspace/src

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

# Copy your new fine-tuned model into the directory defined by REC_MODEL_SUBDIR
COPY PPOCRV4_Default_Dict_Run2/ ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR}/

# Removed the line that copies the custom character dictionary
# COPY custom_char_dict.txt /opt/paddleocr_models/dicts/custom_char_dict.txt

RUN pip install --no-cache-dir -U pip setuptools wheel \
    && pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and create __init__.py to make it a proper Python package
COPY src/ /workspace/src/
RUN touch /workspace/src/__init__.py

# Change working directory to src to fix relative imports
WORKDIR /workspace/src

EXPOSE 5003

# Updated CMD to work with relative imports
CMD ["uvicorn", "ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]