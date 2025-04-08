import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import validate_document, send_email
from pydantic import BaseModel
from utils import predict_grant_category
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Create a FastAPI instance
app = FastAPI(
    title="Alusive Africa ML Solutions",
    description="Interact with Alusive Africa ML Solutions!",
    version="1.0",
)


# Load the pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# faqs
faqs = [
    # Grants
    {"question": "What is Alusive Africa's Tuition Grant Program?",
        "answer": "Alusive Africa's tuition grant program is our third of three pillars to support fee-paying students at The African Leadership University in need."},
    {"question": "When does the next application for Alusive Africa's grant open?",
        "answer": "While we aim to award grants at least twice every year, our applications open on the basis of fund availability."},
    {"question": "Who is eligible to apply for the grant?",
        "answer": "Any student at the African Leadership University in need of some financial support is welcome to apply for Alusive Africa's tuition grant."},
    {"question": "What does the grant cover?",
        "answer": "Alusive Africa tuition grant is only meant to go towards tuition support for students at the African Leadership University."},
    {"question": "What are the criteria for receiving a grant?",
        "answer": "Applicants of Alusive Africa tuition grant program are evaluated on the basis of need, academic performance and community contribution."},
    {"question": "How much can one receive for funding?",
        "answer": "Alusive Africa grant allocations is the result of a holistic evaluation on a case-by-case basis."},
    {"question": "How is the grant amount determined?",
        "answer": "Alusive Africa grant allocation is the result of a thorough holistic evaluation on a case-by-case basis."},
    {"question": "How does one apply for Alusive Africa's Tuition Grant?",
        "answer": "Alusive Africa tuition grant program opens for application whenever we publish, usually via student email."},
    {"question": "How long does the application process take?",
        "answer": "While an application for Alusive Africa's tuition grant program can be done in one sitting, the platform we use allows applicants to pause and resume wherever they left at their convenience."},
    {"question": "What documents do I need to submit with my application?",
        "answer": "Supporting documents for any applicant to demonstrate their financial need, academic performance, and community service vary on a case-by-case basis."},
    {"question": "When will I know the decision regarding my application?",
        "answer": "We endeavor to open applications for Alusive Africa's tuition grant program towards the end of one trimester and announce the decisions during the next trimester."},
    {"question": "Can I apply for the grant if I have received it before?",
        "answer": "Returning applicants are definitely welcome to reapply for Alusive Africa's tuition grant program, even though their priority levels are typically lower."},
    {"question": "What happens if I receive a partial grant and still need more financial support?",
        "answer": "While we would like to support Alusive Africa's grant recipients to the best of our ability, our grant might not entirely cover their tuition fee deficit and we encourage them to explore other avenues of funding."},
    {"question": "Are there any conditions attached to the grant?",
        "answer": "Yes. Any conditions attached to Alusive Africa's tuition grant program are outlined in the application form, and the grant agreement recipients sign in order to accept their offer."},
    {"question": "Can I apply for a grant if I already have other scholarships?",
        "answer": "Yes. Any student in need of financial support towards their tuition at the African Leadership University is welcome to apply for the Alusive Africa tuition grant program."},
    {"question": "What happens if I drop out or defer my studies after receiving a grant?",
        "answer": "Only enrolled students are eligible to apply for Alusive Africa's tuition grant program."},
    {"question": "What is the deadline for grant applications?",
        "answer": "Alusive Africa grant application deadlines are usually clearly communicated when it opens and is also outlined in the application."},
    {"question": "How do I increase my chances of getting a grant?",
        "answer": "Alusive Africa grant allocation is the result of a holistic evaluation on a case-by-case basis so accurate verifiable information to the best of your knowledge is a good place to start."},
    {"question": "Who can I contact if I have issues with my application?",
        "answer": "Feel free to contact Alusive Africa via our email at alusiveafrica.rwa@alustudent.com or phone number +250735545222 about any issues unanswered in our chat."},

    # Internships
    {"question": "What internship opportunities does Alusive Africa offer?",
        "answer": "Alusive Africa internships offer opportunities in Communication, Marketing and Tech."},
    {"question": "Who is eligible to apply for an internship?",
        "answer": "Any enrolled student at the African Leadership University is eligible to apply for Alusive Africa internship."},
    {"question": "How do I apply for an internship?",
        "answer": "Alusive Africa internship applications are usually formally announced via student email."},
    {"question": "Are Alusive Africa internships paid?",
        "answer": "Yes. All Alusive internships are compensated via Alusive credits."},
    {"question": "What are Alusive credits?",
        "answer": "Alusive credits are the payment mode for Alusive internships and are only redeemable toward tuition fees at the African Leadership University."},
    {"question": "What skills or experience do I need to apply?",
        "answer": "As an opportunity to start and grow your career, while Alusive Africa internships would benefit from previous expertise, we prioritize a burning desire to learn on the job."},
    {"question": "How long do internships last?",
        "answer": "Alusive Africa internships last a trimester."},
    {"question": "When do Alusive internships begin?",
        "answer": "Alusive internships usually start at the beginning of a new trimester just after the hiring process is concluded and an agreement accepted by the intern."},
    {"question": "Can interns work remotely?",
        "answer": "Yes. Depending on the role, Alusive internships can be virtual, hybrid or in-person."},
    {"question": "What are the responsibilities of an intern?",
        "answer": "Every role for every internship position at Alusive Africa has different responsibilities clearly outlined in the job description on the internship agreement form."},
    {"question": "Will I receive a certificate or recommendation letter after completing the internship?",
        "answer": "Yes. While Alusive Africa does not provide a certificate after completing the internship, we are more than happy to provide you with a letter of completion and a recommendation letter upon request otherwise a formal email is what we share at the beginning and end of each internship."},
    {"question": "Can an internship lead to a long-term role with Alusive Africa?",
        "answer": "Yes. Alusive Africa internship can definitely lead to a long-term role."},
    {"question": "How competitive is the internship application process?",
        "answer": "Alusive internships are very competitive, given the large pool of applicants."},
    {"question": "What support do interns receive during their internship?",
        "answer": "Alusive interns received robust professional support necessary for their growth during the full period of the internship."},
    {"question": "How many interns does Alusive Africa recruit each cycle?",
        "answer": "The number of interns recruited by Alusive Africa during any internship cycle depends on our need during that period."},
    {"question": "Can I apply if I am a first-year student?",
        "answer": "Yes. All eligible enrolled students are welcome to apply for the Alusive Africa internship irrespective of their year of study."},
    {"question": "What happens if I need to leave the internship early?",
        "answer": "Tendering a notice with your intent to leave should be done 2 weeks in advance for proper team adjustments."},
    {"question": "Do interns get to work on real projects?",
        "answer": "Yes. All Alusive Africa internship projects are real-world and consequential."},
    {"question": "Is there any mentorship provided during the internship?",
        "answer": "Yes. Career development through mentorship is a core part of the Alusive Africa internship."},
    {"question": "Can I apply for both an internship and a grant at the same time?",
        "answer": "Yes. Eligible candidates are welcome to apply for both the internship and grant at Alusive Africa."},
    {"question": "Do interns receive any training or onboarding?",
        "answer": "Yes. A comprehensive training and onboarding process is primary for every cycle of Alusive Africa internship."},

    # Student Venture Support
    {"question": "What kind of support does Alusive Africa offer to student entrepreneurs?",
        "answer": "At this time, Alusive Africa offers support through meaningful partnerships with student ventures with whom we share goals."},
    {"question": "Who is eligible for student venture support?",
        "answer": "All enrolled students interested in Alusive student venture support are welcome to apply by sending their proposal to Alusive Africa via email."},
    {"question": "Does Alusive Africa provide funding for student startups?",
        "answer": "No. At this time, our funding model has yet to directly extend cash support to student startups."},
    {"question": "How can I apply for venture support?",
        "answer": "Writing a proposal to us at Alusive Africa through our email address is how to apply for student venture support."},
    {"question": "What types of businesses does Alusive Africa support?",
        "answer": "Alusive Africa supports all business ventures that pass our evaluation for meaningful partnership."},
    {"question": "Do I need to be part of a team to receive support?",
        "answer": "No. You don't need to be part of a team to receive support."},
    {"question": "Can I apply if my venture is still in the idea stage?",
        "answer": "Yes. Eligible students are welcome to apply no matter the stage they are in their business."},
    {"question": "Are there networking opportunities for student founders?",
        "answer": "Yes. Plenty of networking opportunities exist for student founders who partner with Alusive Africa."},
    {"question": "Does Alusive Africa take any equity in student startups?",
        "answer": "Not yet. Our partnership model is yet to explore taking a stake in terms of equity in student startups."},
    {"question": "Can I apply for both a grant and venture support?",
        "answer": "Yes. Eligible applicants are more than welcome to apply for both Alusive grants and our student venture support."},

    # General
    {"question": "What is Alusive Africa?", "answer": "Alusive Africa is a student-led non-profit organization based at the African Leadership University on a mission to support fee-paying students with our grants, offer career development opportunities through our internships and invest in student-led ventures."},
    {"question": "What does Alusive Africa do?", "answer": "Alusive Africa facilitates grant-based tuition support, contributes towards student career development opportunities and collaborates with other student-led ventures to foster community development through initiatives like the Giveaway4Good."},
    {"question": "How does Alusive Africa raise funds for grants?",
        "answer": "Alusive Africa raises its funding through donations, partnerships, and fundraising initiatives. The organization is also working toward establishing 501(c)(3) status in the U.S. to expand its fundraising capabilities."},
    {"question": "What is the Giveaway4Good initiative?",
        "answer": "The Giveaway4Good is an initiative under Alusive Africa that partners with other student-led ventures to equip, enable and empower fellow students, emerging talents and aspiring founders."},
    {"question": "How can I support Alusive Africa?",
        "answer": "You can support Alusive Africa by Donating to the grant fund, Partnering with the organization, Volunteering or contributing skills, and Spreading awareness about its mission."},
    {"question": "Is Alusive Africa affiliated with African Leadership University (ALU)?", "answer": "Yes. While Alusive Africa currently operates within the African Leadership University and primarily supports its students, it is an independent initiative working toward becoming a legally registered non-profit entity while still investing in our relationship with the African Leadership University as a founding partner."},
    {"question": "What is the long-term vision of Alusive Africa?", "answer": "The long-term vision is to establish Alusive Africa as a fully independent legal entity, expand its impact beyond ALU, and create a sustainable, student-led structure that continues to provide financial support, career development opportunities and venture support to students across Africa."},
    {"question": "How can I contact Alusive Africa?",
        "answer": "Alusive Africa can be contacted via email at alusiveafrica.rwa@alustudent.com or phone number at +250735545222."},
]


# Feature columns (ensure this matches the feature columns used in your training data)
FEATURE_COLUMNS = [
    "Fee balance (USD)",
    "Total Monthly Income",
    "Students in Household",
    "Household Size",
    "Household Supporters",
    "Household Dependants",
    "ALU Grant Amount",
    "Grant Requested",
    "Amount Affordable",
    "fee_to_income",
    "dependants_per_supporter",
    "requested_to_affordable",
    "household_income_per_person",
    "Academic Standing_No",
    "Academic Standing_Yes",
    "Disciplinary Standing_No",
    "Disciplinary Standing_Yes",
    "Financial Standing_No",
    "Financial Standing_Yes",
    "ALU Grant Status_No",
    "ALU Grant Status_Yes",
    "Previous Alusive Grant Status_No",
    "Previous Alusive Grant Status_Yes",
]


# Define expected request body
# Define the input data model for the applicant
class ApplicantData(BaseModel):
    academic_standing: str
    disciplinary_standing: str
    financial_standing: str
    alu_grant_status: str
    previous_alusive_grant: str
    fee_balance: float
    total_monthly_income: float
    students_in_household: int
    household_size: int
    household_supporters: int
    household_dependants: int
    alu_grant_amount: float
    grant_requested: float
    amount_affordable: float


# Allow CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Email & Notification message templates
def generate_messages(first_name, last_name, document_type, status):
    """Generates well-formatted notification and email messages based on the document status."""

    if status == "signed":
        notification = (
            f"Dear {first_name},\n\n"
            f"Thank you for signing your '{document_type}'.\n\n"
            f"You will receive an email shortly once it has been verified."
        )

        email = (
            f"Dear {first_name} {last_name},\n\n"
            f"Alusive Africa is delighted to acknowledge receipt of a signed copy of your '{document_type}'.\n\n"
            f"Thank you."
        )

    else:  # If the document is unsigned
        notification = (
            f"Dear {first_name},\n\n"
            f"The document you uploaded is unsigned.\n\n"
            f"Please sign the document and upload it again."
        )

        email = (
            f"Dear {first_name} {last_name},\n\n"
            f"We noticed that the '{document_type}' you uploaded is unsigned.\n\n"
            f"Kindly sign it and re-submit.\n\n"
            f"Thank you."
        )

    return {"notification": notification, "email": email}


@app.post("/validate/")
async def validate_file(
    document: UploadFile = File(...),
    full_name: str = Form(...),
    email: str = Form(...),
    document_type: str = Form(...),
):
    """
    Handles document verification.

    - Accepts a document file and metadata (`full_name`, `email`, `document_type`).
    - Validates document type (`grant` or `internship`).
    - Uses the `validate_document` function from `utils.py`.
    - Returns analysis results with metadata and personalized messages.
    """
    # Validate document type
    if document_type not in ["grant", "internship"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid document type. Must be 'grant' or 'internship'.",
        )

    # Extract first and last name
    name_parts = full_name.strip().split()
    first_name = name_parts[0]  # First word is the first name
    last_name = (
        " ".join(name_parts[1:]) if len(name_parts) > 1 else ""
    )  # Rest is last name

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(document.filename)[-1]
    ) as temp_file:
        temp_file.write(await document.read())
        temp_path = temp_file.name

    try:
        # Perform document validation
        result = validate_document(temp_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )
    finally:
        os.remove(temp_path)  # Cleanup temp file

    # Generate messages based on validation result
    document_status = result["prediction"]  # "signed" or "unsigned"
    messages = generate_messages(
        first_name, last_name, document_type, document_status)

    # Send email notification
    send_email(email, messages["email"])

    # Prepare response
    response_data = {
        "full_name": full_name,
        "email": email,
        "document_type": document_type,
        "result": result,
        "notification": messages["notification"],
    }
    return JSONResponse(content=response_data)


# Define the API endpoint to predict the grant category
@app.post("/predict-grant/")
def predict_grant(applicant: ApplicantData):
    """
    API endpoint to predict the grant category based on the applicant's data.
    """
    try:
        # Convert input data to dictionary
        applicant_data = {
            "Academic Standing": applicant.academic_standing,
            "Disciplinary Standing": applicant.disciplinary_standing,
            "Financial Standing": applicant.financial_standing,
            "ALU Grant Status": applicant.alu_grant_status,
            "Previous Alusive Grant Status": applicant.previous_alusive_grant,
            "Fee balance (USD)": applicant.fee_balance,
            "Total Monthly Income": applicant.total_monthly_income,
            "Students in Household": applicant.students_in_household,
            "Household Size": applicant.household_size,
            "Household Supporters": applicant.household_supporters,
            "Household Dependants": applicant.household_dependants,
            "ALU Grant Amount": applicant.alu_grant_amount,
            "Grant Requested": applicant.grant_requested,
            "Amount Affordable": applicant.amount_affordable,
        }

        # Call the predict_grant_category function from utils.py
        result = predict_grant_category(applicant_data, FEATURE_COLUMNS)

        # Return the result of the prediction
        return result

    except Exception as e:
        # Return an HTTP error with a message if something goes wrong
        raise HTTPException(status_code=500, detail=str(e))


# Precompute embeddings
faq_questions = [faq["question"] for faq in faqs]
faq_embeddings = model.encode(faq_questions, convert_to_tensor=True)


# Request model for API
class QuestionRequest(BaseModel):
    question: str


# Function to get the best answer
def get_answer(user_question: str):
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    similarities = util.cos_sim(user_embedding, faq_embeddings)
    best_match_idx = np.argmax(similarities)
    similarity_score = similarities[0][best_match_idx].item()

    # Set a threshold to filter out low-confidence responses
    if similarity_score > 0.6:  # Adjust as needed
        return {
            "answer": faqs[best_match_idx]["answer"],
            "confidence": similarity_score,
        }
    else:
        return {
            "answer": "Sorry, I couldn't find a good match for your question. Try rephrasing it or contact us at alusiveafrica.rwa@alustudent.com for help.",
            "confidence": similarity_score,
        }

# Define API endpoint


@app.post("/chat/", summary="Ask the chatbot a question")
async def chat(request: QuestionRequest):
    return get_answer(request.question)

# Root endpoint


@app.get("/", summary="Root Endpoint")
async def root():
    return {"message": "Welcome to Alusive Africa ML Solutions. Enjoy our services."}
