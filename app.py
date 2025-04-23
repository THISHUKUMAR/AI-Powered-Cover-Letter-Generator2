import streamlit as st
from backend import load_pdf_text, create_vectorstore, generate_cover_letter

st.set_page_config(page_title="AI Cover Letter Generator", page_icon="📄", layout="centered")

st.title("📄 AI-Powered Cover Letter Generator")
st.write("Upload your **resume** and **job description** to generate a personalized cover letter.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])

if resume_file and jd_file:
    if st.button("✨ Generate Cover Letter"):
        with st.spinner("Processing and generating..."):
            resume_text = load_pdf_text(resume_file)
            jd_text = load_pdf_text(jd_file)

            vectorstore = create_vectorstore(resume_text, jd_text)

            # Generate the cover letter (returns plain text)
            cover_letter = generate_cover_letter(vectorstore)

        st.success("✅ Cover letter generated successfully!")
        
        st.subheader("📄 Generated Cover Letter")
        # Display it nicely using markdown
        st.markdown(f"```\n{cover_letter}\n```")

        st.download_button(
            label="📥 Download Cover Letter",
            data=cover_letter,
            file_name="cover_letter.txt",
            mime="text/plain"
        )
else:
    st.warning("Please upload both your resume and job description.")
