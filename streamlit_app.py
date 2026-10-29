import json
from pathlib import Path
from uuid import uuid4

import streamlit as st


st.set_page_config(page_title="Video Analysis", page_icon="🎥")
st.title("Video Analysis")
st.write("Upload a video and select its spoken language.")


def clear_results():
    st.session_state.pop("analysis_result", None)


language_label = st.selectbox(
    "Video language", ["Sinhala", "English"], on_change=clear_results
)
language = {"Sinhala": "si", "English": "en"}[language_label]
uploaded_video = st.file_uploader(
    "Choose an MP4 video", type=["mp4"], max_upload_size=200,
    on_change=clear_results,
)

if uploaded_video is None:
    st.info("Choose a video to begin.")
    st.stop()

st.write("File:", uploaded_video.name)
st.write("Selected language:", language_label)
st.video(uploaded_video.getvalue())

if st.button("Process video", type="primary"):
    clear_results()
    progress = st.progress(0)
    with st.status("Preparing video…", expanded=True) as status:
        try:
            from app.services.upload_pipeline import process_video

            run_dir = Path(__file__).resolve().parent / "data" / "upload_runs" / uuid4().hex
            run_dir.mkdir(parents=True, exist_ok=False)
            video_path = run_dir / "input.mp4"
            video_path.write_bytes(uploaded_video.getvalue())

            def update_progress(step, label):
                status.update(label=label + "…")
                st.write(label)
                progress.progress((step - 1) / 8)

            result = process_video(
                video_path, language, run_dir / "results",
                on_progress=update_progress,
                original_name=uploaded_video.name,
            )
            st.session_state["analysis_result"] = result
            progress.progress(1.0)
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Processing failed", state="error", expanded=True)
            st.error("The video could not be processed. See the details below.")
            with st.expander("Error details"):
                st.exception(exc)

result = st.session_state.get("analysis_result")
if result is not None:
    st.caption("Candidate claims and search drafts only. Evidence has not been retrieved and claims have not been fact-checked.")
    st.subheader("Transcript")
    st.write(result["text"])
    st.subheader("Summary")
    st.write(result["summary"])
    claims = result["claim_analysis"]["claims"]
    st.subheader(f"Candidate claims ({len(claims)})")
    for claim in claims:
        with st.expander(f"Claim {claim['claim_id']}"):
            st.write(claim["text"])
            st.write("Check-worthiness score:", claim["check_worthy_score"])
            st.caption("This score is not a probability that the claim is true.")
            screening = claim.get("review_screening_status", "not_performed")
            if claim.get("needs_review"):
                st.warning("Needs review: " + "; ".join(claim.get("review_reasons", [])))
            elif screening == "not_performed":
                st.info("Review screening was not performed for this claim.")
            else:
                st.info("No configured review cue matched. The claim remains unverified.")
            st.write("Named entities")
            st.json(claim.get("entities", {}))
            st.write("Search drafts")
            drafts = claim.get("search_queries", {})
            for query in drafts.get("queries", []):
                st.write(query["query"])
    st.download_button(
        "Download JSON results",
        data=json.dumps(result, ensure_ascii=False, indent=4),
        file_name="video_analysis.json",
        mime="application/json",
        on_click="ignore",
    )
    st.caption("This run’s video, audio, frames, transcript, and JSON are saved under data/upload_runs on your computer.")
