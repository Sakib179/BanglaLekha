"""Run: python -m streamlit run app.py"""
import hashlib
import json
import logging
import os
import threading
from html import escape

os.environ.setdefault('KERAS_BACKEND', 'torch')
os.environ.setdefault('ONEDNN_PRIMITIVE_CACHE_CAPACITY', '4')

import streamlit as st
import streamlit.components.v1 as components
from ocr_service import ROOT, canvas_image, model_paths, read_image

st.set_page_config(page_title='Lipi · Bangla handwriting', page_icon='✍', layout='wide')
st.markdown(f"<style>{(ROOT / 'ui/style.css').read_text()}</style>", unsafe_allow_html=True)
draw_word = components.declare_component('lipi_canvas', path=str(ROOT / 'ui/canvas'))


@st.cache_resource(show_spinner=False)
def load_models(checkpoint, helper, model_version):
    # Cache frozen weights. A lock protects the shared engine across browser sessions.
    import torch
    from hybrid_ocr import load_hybrid_engine
    torch.set_num_threads(2)
    return load_hybrid_engine(checkpoint, helper_dir=helper), threading.Lock()


def recognize(image, source):
    checkpoint, helper = model_paths()
    try:
        required = [checkpoint, helper / 'model.h5', helper / 'class_map.csv']
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError('Missing model assets. See README model setup. ' + ', '.join(missing))
        version = tuple(p.stat().st_mtime_ns for p in required)
        digest = hashlib.sha256(str(image.size).encode() + image.tobytes() + str(version).encode()).hexdigest()
        if st.session_state.get('image_digest') != digest:
            with st.spinner('Reading your handwriting…'):
                engine, lock = load_models(str(checkpoint), str(helper), version)
                from hybrid_ocr import predict_word
                with lock:
                    result = predict_word(engine, image)
                st.session_state.result = {k: v for k, v in result.items() if k != 'debug'}
            st.session_state.image_digest = digest
        st.session_state.source = source
        st.session_state.pop('recognition_error', None)
    except (OSError, ValueError, RuntimeError) as error:
        logging.exception('Word recognition failed')
        st.session_state.pop('result', None)
        st.session_state.pop('image_digest', None)
        st.session_state.recognition_error = str(error)


def render_result():
    result = st.session_state.get('result')
    if st.session_state.get('recognition_error'):
        st.error(st.session_state.recognition_error)
    if not result:
        st.markdown('''<div class="result-card empty-result"><div class="result-symbol">অ</div>
            <span class="eyebrow">A LITTLE INK. A WHOLE WORD.</span><h3>Your word appears here.</h3>
            <p>Write on the canvas or upload a handwritten word.<br>We’ll turn the strokes into Bangla text.</p>
            </div>''', unsafe_allow_html=True)
        return
    uncertain = result['status'] != 'accepted'
    badge = 'Review suggested' if uncertain else 'Prediction ready'
    color = 'review' if uncertain else 'ready'
    st.markdown(f'''<div class="result-card"><div class="result-top">
        <span class="eyebrow">PREDICTED WORD</span><span class="result-badge {color}">{badge}</span></div>
        <div class="predicted-word" lang="bn">{escape(result['word'])}</div>
        <div class="result-meta">Last submitted · {escape(st.session_state.source)}
        <span>{result['latency_ms'] / 1000:.2f}s</span></div></div>''', unsafe_allow_html=True)
    st.caption('Some strokes were ambiguous. Check the word or try clearer spacing.' if uncertain
               else 'A model prediction, not a verified transcription. Please check the word.')
    st.download_button('↓  Save word', result['word'], 'bangla_word.txt', 'text/plain', use_container_width=True)
    with st.expander('Recognition details'):
        st.write('Your ResNet-18 reads base characters. A frozen Bengali.AI helper reads vowel signs.')
        rows = [{'Base': t['base'], 'Vowel sign': t['vowel'] or '—',
                 'Base score': round(t['base_probability'], 3),
                 'Vowel score': round(t['vowel_probability'], 3)} for t in result['tokens']]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        st.caption('Scores are uncalibrated; they do not measure word accuracy.')
        st.download_button('Download prediction report', json.dumps(result, ensure_ascii=False, indent=2),
                           'prediction.json', 'application/json')


st.markdown('''<div class="topbar"><div class="brand"><span class="brand-mark">লি</span>
    <span>lipi<span class="brand-dot">.</span></span><span class="brand-divider"></span>
    <span class="brand-subtitle">BANGLA HANDWRITING STUDIO</span></div>
    <span class="model-pill"><span class="green-dot"></span>ResNet-18 · BanglaLekha</span></div>
    <div class="hero"><div><span class="eyebrow">FROM PEN TO POSSIBILITY</span>
    <h1>Your handwriting.<br><span>In words.</span></h1>
    <p>A space for your Bangla handwriting.<br>Draw a word. Let the ink speak.</p></div>
    <div class="hero-art" aria-hidden="true"><span>বাংলা</span><small>EVERY STROKE TELLS A STORY</small></div></div>
    ''', unsafe_allow_html=True)

left, right = st.columns([1.45, 1], gap='large')
with left:
    st.markdown('<div class="section-heading"><span class="step-number">01</span><h2>Make your mark</h2></div>', unsafe_allow_html=True)
    draw_tab, upload_tab = st.tabs(['✎  Draw a word', '↑  Upload an image'])
    with draw_tab:
        drawing = draw_word(key='word_canvas', default=None)
        if drawing and drawing.get('request_id') != st.session_state.get('drawing_request'):
            st.session_state.drawing_request = drawing['request_id']
            try:
                recognize(canvas_image(drawing['image']), 'Drawing canvas')
            except ValueError as error:
                st.session_state.pop('result', None)
                st.session_state.recognition_error = str(error)
    with upload_tab:
        uploaded = st.file_uploader('Upload one handwritten Bangla word', type=['png', 'jpg', 'jpeg', 'bmp', 'webp'])
        if uploaded is not None:
            try:
                image = read_image(uploaded.getvalue())
                st.image(image, caption=uploaded.name, use_container_width=True)
                if st.button('Recognize image  →', type='primary', use_container_width=True):
                    recognize(image, uploaded.name)
            except ValueError as error:
                st.error(str(error))
    st.markdown('<p class="input-tip">A little room helps. Keep one whole word in view, including its vowel signs.</p>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-heading"><span class="step-number">02</span><h2>Read the result</h2></div>', unsafe_allow_html=True)
    render_result()

st.markdown('''<div class="footer"><span>Built with care for the language we write.</span>
    <span>Sakib Rayhan Yeasin <span class="footer-dot">·</span> 1000128462</span></div>''', unsafe_allow_html=True)
