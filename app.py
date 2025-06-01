from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_pymongo import PyMongo
import re
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
import numpy as np
from bson import ObjectId
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from flask import send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from flask import current_app
from reportlab.platypus import Image
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer,BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet








app = Flask(__name__)
app.secret_key = 'your_secret_key'


app.config["MONGO_URI"] = "mongodb+srv://niyacv13:niyaviju@cluster0.78z9j.mongodb.net/fetalstatus?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)
db = mongo.db  


if db is None:
    raise Exception("MongoDB connection failed!")


admin_collection = db.admin_users
users_collection = db.users  
lab_reports_collection = db["lab_reports"]





if not admin_collection.find_one({"email": "fhadmin@gmail.com"}):
    admin_collection.insert_one({"email": "fhadmin@gmail.com", "password": "admin123"})



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')





@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form.get("email")  
        password = request.form.get("password")

        admin = admin_collection.find_one({"email": email, "password": password})  
        if admin:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials", "error")

    return render_template("admin_login.html")



@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect(url_for("admin_login"))
    return render_template("admin_dashboard.html")


@app.route("/fetch_users/<role>")
def fetch_users(role):
    if "admin" not in session:
        return jsonify([])

    users = users_collection.find({"role": role})
    users_list = []

    for user in users:
        user_id = user.get("lab_assistant_id") if role == "lab_assistant" else user.get("doctor_id")
        updated_at = user.get("updated_at", "N/A")

        if updated_at != "N/A":
            date_part, time_part = updated_at.split(" ")
        else:
            date_part, time_part = "N/A", "N/A"

        users_list.append({
            "user_id": user_id,
            "_id": str(user["_id"]),  
            "username": user["username"],
            "email": user["email"],
            "password": user["password"],
            "status": user.get("status", "pending"),
            "updated_date": date_part,
            "updated_time": time_part,
            "mongo_id": user["_id"]  
        })

    status_order = {"pending": 0, "approved": 1, "rejected": 2}

    users_list.sort(key=lambda x: (
        status_order[x["status"]],          
        str(x["mongo_id"]) if x["status"] == "pending" else "",  
        x["updated_date"] if x["status"] != "pending" else ""    
    ), reverse=False)

    return jsonify(users_list)


@app.route("/update_status", methods=["POST"])
def update_status():
    if "admin" not in session:
        return jsonify({"message": "Unauthorized"}), 401

    data = request.json
    user_id = data["userId"]
    status = data["status"]
    update_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if user:
        users_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"status": status, "updated_at": update_time}}
        )
        return jsonify({"message": f"User {status} successfully!", "role": user["role"], "updated_at": update_time})

    return jsonify({"message": "User not found"}), 404



@app.route("/delete_user", methods=["POST"])
def delete_user():
    if "admin" not in session:
        return jsonify({"message": "Unauthorized"}), 401

    data = request.json
    user_id = data["userId"]

    result = users_collection.delete_one({"_id": ObjectId(user_id)})
    if result.deleted_count > 0:
        return jsonify({"message": "User deleted successfully!"})

    return jsonify({"message": "User not found"}), 404






@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        role = request.form.get("role")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        email = request.form.get("email")
        phone = request.form.get("phone")
        lab_assistant_id = request.form.get("lab_assistant_id")
        doctor_id = request.form.get("doctor_id")

        
        lab_assistant_pattern = r"^FHL([1-5][0-9]{2}|600)$"
        doctor_pattern = r"^FHD([1-5][0-9]{2}|600)$"

        if role == "lab_assistant":
            if not lab_assistant_id or not re.match(lab_assistant_pattern, lab_assistant_id):
                flash("Invalid Lab Assistant ID!", "error")
                return redirect(url_for("register"))

        if role == "doctor":
            if not doctor_id or not re.match(doctor_pattern, doctor_id):
                flash("Invalid Doctor ID!", "error")
                return redirect(url_for("register"))

        if role in ["lab_assistant", "doctor"]:
            status = "pending"
        else:
            status = "approved"

        
        if users_collection.find_one({"username": username}):
            flash("Username already exists!", "error")
            return redirect(url_for("register"))

        if users_collection.find_one({"email": email}):
            flash("Email ID already exists!", "error")
            return redirect(url_for("register"))

        if users_collection.find_one({"phone": phone}):
            flash("Phone number already exists!", "error")
            return redirect(url_for("register"))

        if role == "lab_assistant" and users_collection.find_one({"lab_assistant_id": lab_assistant_id}):
            flash("Lab Assistant ID already exists!", "error")
            return redirect(url_for("register"))

        if role == "doctor" and users_collection.find_one({"doctor_id": doctor_id}):
            flash("Doctor ID already exists!", "error")
            return redirect(url_for("register"))

        
        strong_password_regex = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{6,}$"
        if not re.match(strong_password_regex, password):
            flash("Weak password! Use uppercase, lowercase, number, and special character.", "error")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match", "error")
            return redirect(url_for("register"))

        
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            flash("Invalid email format", "error")
            return redirect(url_for("register"))

        
        if not re.match(r"^\d{10}$", phone):
            flash("Phone number must be 10 digits", "error")
            return redirect(url_for("register"))

       
        user_data = {
            "role": role,
            "username": username,
            "password": password,
            "email": email,
            "phone": phone,
            "status": status
        }

        if role == "lab_assistant":
            user_data["lab_assistant_id"] = lab_assistant_id
        elif role == "doctor":
            user_data["doctor_id"] = doctor_id

        users_collection.insert_one(user_data)

        if status == "approved":
            flash("Registration successful, You can now log in.", "success")
        else:
            flash("Registration successful , Login to see admin status.", "info")

        return redirect(url_for("register"))

    return render_template("register.html")






@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form.get("role")
        email = request.form.get("email")
        password = request.form.get("password")
        doctor_id = request.form.get("doctor_id")
        lab_assistant_id = request.form.get("lab_assistant_id")

        user = users_collection.find_one({"email": email})

        if not user:
            flash("User Does Not Exist!", "error")
            return redirect(url_for("login"))

        if user["role"] != role:
            flash("Invalid Role!", "error")
            return redirect(url_for("login"))

        if user["password"] != password:
            flash("Invalid Password!", "error")
            return redirect(url_for("login"))

        if user["status"] == "pending":
            flash("Your account is awaiting admin approval.", "error")
            return redirect(url_for("login"))

        if user["status"] == "rejected":
            flash("Admin has rejected your account.", "error")
            return redirect(url_for("login"))

        
        session["user_id"] = str(user["_id"])
        session["username"] = user["username"]
        session["role"] = user["role"]

        
        if role == "patient":

            return redirect(url_for("patient_dashboard"))

        elif role == "doctor":
            if not doctor_id or doctor_id != user.get("doctor_id"):
                flash("Invalid Doctor ID!", "error")
                return redirect(url_for("login"))

            return redirect(url_for("doctor_dashboard"))

        elif role == "lab_assistant":
            if not lab_assistant_id or lab_assistant_id != user.get("lab_assistant_id"):
                flash("Invalid Lab Assistant ID!", "error")
                return redirect(url_for("login"))

            return redirect(url_for("lab_assistant_dashboard"))

        flash("User Does Not Exist!", "error")
        return redirect(url_for("login"))

    return render_template("login.html")



@app.route("/patient_dashboard")
def patient_dashboard():
    if "username" not in session or session["role"] != "patient":
        return redirect(url_for("login"))

    patient_username = session["username"]

    
    lab_report = lab_reports_collection.find_one(
        {"patient_username": patient_username},
        sort=[("date", -1), ("time", -1)]  
    )

    if lab_report:
        
        lab_report["_id"] = str(lab_report["_id"])

    return render_template("patient_dashboard.html", username=patient_username, lab_report=lab_report)





@app.route("/download_report")
def download_report():
    if "username" not in session or session["role"] != "patient":
        return redirect(url_for("login"))

    patient_username = session["username"]

    lab_report = lab_reports_collection.find_one(
        {"patient_username": patient_username},
        sort=[("date", -1), ("time", -1)]
    )

    if not lab_report:
        return "No report found!", 404

    doctor_name = request.args.get("doctor_name", "Dr. Nisa")

    
    def draw_page_border(canvas, doc):
        width, height = A4
        margin = 36  
        canvas.saveState()
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(2)
        canvas.rect(margin, margin, width - 2 * margin, height - 2 * margin)
        canvas.restoreState()

    pdf_buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    elements = []

    
    title = Paragraph("<b>FETOHEALTH REPORT</b>", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))

    
    patient_info = f"""
    <b>Patient Details:</b><br/>
    Name: {lab_report['patient_username']}<br/>
    Email: {lab_report['patient_email']}<br/>
    Phone: {lab_report['patient_phone']}<br/><br/>
    
    <b>Medical Staff:</b><br/>
    Lab Assistant: {lab_report['lab_assistant_username']}<br/>
    Doctor: {doctor_name}<br/>
    <b>Report Generated On:</b><br/>
    Date: {lab_report['date']}<br/>
    Time: {lab_report['time']}<br/>
    """
    elements.append(Paragraph(patient_info, styles['Normal']))
    elements.append(Spacer(1, 12))

    
    prediction = f"<b>Prediction Result:</b><br/>Fetal Health Condition: {lab_report['prediction']}"
    elements.append(Paragraph(prediction, styles['Normal']))
    elements.append(Spacer(1, 12))

    
    elements.append(Paragraph("<b>Test Values:</b>", styles['Heading3']))
    table_data = [["Feature", "Value"]]
    table_data += [[key, str(value)] for key, value in lab_report['values'].items()]
    table = Table(table_data, colWidths=[270, 230])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 24))

    
    elements.append(Paragraph("<b>Doctor's Observation:</b>", styles['Heading3']))
    elements.append(Paragraph(lab_report['doctor_message'], styles['Normal']))

    
    doc = BaseDocTemplate(pdf_buffer, pagesize=A4)
    frame = Frame(36, 36, A4[0] - 72, A4[1] - 72, id='normal')
    doc.addPageTemplates([PageTemplate(id='bordered_page', frames=frame, onPage=draw_page_border)])
    doc.build(elements)

    pdf_buffer.seek(0)
    return send_file(pdf_buffer, as_attachment=True, download_name="Fetal_Health_Report.pdf", mimetype="application/pdf")




@app.route('/lab_assistant_dashboard', methods=['GET', 'POST'])
def lab_assistant_dashboard():
    if "username" not in session or session["role"] != "lab_assistant":
        return redirect(url_for("login"))

    lab_reports_collection = mongo.db.lab_reports
    users_collection = mongo.db.users

    
    existing_patients = {report["patient_username"] for report in lab_reports_collection.find()}
    new_patients = users_collection.find({"role": "patient", "username": {"$nin": list(existing_patients)}})

    
    for patient in new_patients:
        lab_reports_collection.insert_one({
            "patient_username": patient["username"],
            "patient_email": patient["email"],
            "patient_phone": patient["phone"],
            "lab_assistant_username": session["username"],  
            "date": "",
            "time": "",
            "status": "",
            "values": None
        })

    
    sort_option = request.args.get("sort", "new_to_old") 

    query = {"lab_assistant_username": session["username"]}

    if sort_option == "pending":
        query["status"] = "Pending"
    elif sort_option == "predicted":
        query["status"] = "Predicted"

    
    sort_order = [
        ("status", 1),  
        ("date", -1),
        ("time", -1)
    ]

    if sort_option == "old_to_new":
        sort_order = [
            ("status", 1),  
            ("date", 1),
            ("time", 1)
        ]

    
    reports = list(lab_reports_collection.find(query).sort(sort_order))

    return render_template("lab_assistant_dashboard.html", reports=reports, selected_sort=sort_option)




@app.route('/enter_values/<report_id>', methods=['GET', 'POST'])
def enter_values(report_id):
    if "username" not in session or session["role"] != "lab_assistant":
        return redirect(url_for("login"))

    lab_reports_collection = mongo.db.lab_reports
    report = lab_reports_collection.find_one({"_id": ObjectId(report_id)})

    if request.method == "POST":
        test_values = {field: request.form[field] for field in [
            'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
            'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
            'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
            'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
            'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
            'histogram_variance', 'histogram_tendency'
        ]}

        updated_date = datetime.now().strftime("%Y-%m-%d")
        updated_time = datetime.now().strftime("%H:%M:%S")

        lab_reports_collection.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {
                "values": test_values,
                "date": updated_date,
                "time": updated_time,
                "status": "Pending"
            }}
        )

        return redirect(url_for("lab_assistant_dashboard"))

    return render_template("enter_values.html", patient=report)


@app.route('/delete_report/<report_id>', methods=['POST'])
def delete_report(report_id):
    if "username" not in session or session["role"] != "lab_assistant":
        return redirect(url_for("login"))

    lab_reports_collection = mongo.db.lab_reports

    
    lab_reports_collection.update_one(
        {"_id": ObjectId(report_id)},
        {"$set": {"lab_assistant_username": None}}
    )

    return redirect(url_for("lab_assistant_dashboard"))








with open("ctg_ensemble_model.pkl", "rb") as model_file:
    ensemble_model = pickle.load(model_file)
with open("ctg_scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
with open("ctg_shap_explainer.pkl", "rb") as shap_file:
    loaded_explainer = pickle.load(shap_file)


class_labels = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}


@app.route('/doctor_dashboard')
def doctor_dashboard():
    if "username" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    lab_reports_collection = mongo.db.lab_reports

    
    sort_option = request.args.get("sort", "new_to_old")  

    query = {}  

    
    if sort_option == "pending":
        query["status"] = "Pending"
    elif sort_option == "predicted":
        query["status"] = "Predicted"

    
    sort_order = [("date", -1), ("time", -1)]  
    if sort_option == "old_to_new":
        sort_order = [("date", 1), ("time", 1)]

    
    reports = list(lab_reports_collection.find(query).sort(sort_order))

    return render_template("doctor_dashboard.html", reports=reports, selected_sort=sort_option)



@app.route('/view_values/<report_id>')
def view_values(report_id):
    if "username" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    lab_reports_collection = mongo.db.lab_reports
    users_collection = mongo.db.users  

    report = lab_reports_collection.find_one({'_id': ObjectId(report_id)})

    if not report:
        flash("Report not found.", "danger")
        return redirect(url_for("doctor_dashboard"))

    
    patient = users_collection.find_one({'username': report.get('patient_username')})
    lab_assistant = users_collection.find_one({'username': report.get('lab_assistant_username')})

    return render_template("view_values.html",
                           report=report,
                           patient=patient,
                           lab_assistant=lab_assistant)



@app.route('/predict/<report_id>', methods=['POST'])
def predict(report_id):
    try:
        if "username" not in session or session["role"] != "doctor":
            return redirect(url_for("login"))

        lab_reports_collection = mongo.db.lab_reports
        report = lab_reports_collection.find_one({'_id': ObjectId(report_id)})

        if not report or 'values' not in report:
            return redirect(url_for('doctor_dashboard'))

        
        lab_reports_collection.update_one(
            {'_id': ObjectId(report_id)},
            {'$set': {'status': 'Predicted'}}
        )

        
        test_values = report['values']
        user_input = [float(test_values[key]) for key in test_values.keys()]

        columns = [
            'baseline value', 'accelerations', 'fetal_movement', 'uterine_contractions',
            'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
            'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability',
            'histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
            'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean', 'histogram_median',
            'histogram_variance', 'histogram_tendency'
        ]

        new_data = pd.DataFrame([user_input], columns=columns)
        new_data_scaled = scaler.transform(new_data)

        prediction = ensemble_model.predict(new_data_scaled)[0]
        predicted_class = prediction + 1
        prediction_label = class_labels.get(predicted_class, "Unknown")

        prediction_probabilities = ensemble_model.predict_proba(new_data_scaled)[0].tolist()

        
        shap_values = loaded_explainer.shap_values(new_data_scaled)
        mean_shap_values = np.abs(np.mean(shap_values, axis=0))
        shap_importance = sorted(zip(columns, mean_shap_values), key=lambda x: x[1], reverse=True)
        top_3_features = shap_importance[:3]

        
        shap_importance_df = pd.DataFrame(shap_importance, columns=["Feature", "SHAP Importance"])
        top_3_df = pd.DataFrame(top_3_features, columns=["Feature", "SHAP Importance"])

        
        plt.figure(figsize=(10, 6))
        plt.barh(shap_importance_df["Feature"][::-1], shap_importance_df["SHAP Importance"].values[::-1], color="skyblue")
        plt.xlabel("SHAP Value")
        plt.ylabel("Feature")
        plt.title("SHAP Feature Importance For All Features")
        shap_all_path = os.path.join("static", "shap_all_features.png")
        plt.savefig(shap_all_path, bbox_inches='tight')
        plt.close()

        
        plt.figure(figsize=(6, 4))
        plt.barh(top_3_df["Feature"][::-1], top_3_df["SHAP Importance"].values[::-1], color="green")
        plt.xlabel("SHAP Value")
        plt.ylabel("Feature")
        plt.title("Top 3 SHAP Feature Importance")
        shap_top3_path = os.path.join("static", "shap_top3_features.png")
        plt.savefig(shap_top3_path, bbox_inches='tight')
        plt.close()

        
        
        
        feature_explanations = {
            "baseline_value": {
                "Normal": "Baseline heart rate suggests stable autonomic function and adequate fetal oxygenation.",
                "Suspect": "Slight irregularities in baseline may reflect transient maternal or fetal stressors.",
                "Pathological": "Abnormal baseline indicates possible fetal hypoxia or compromised neurological status."
            },
            "accelerations": {
                "Normal": "Consistent accelerations indicate intact fetal autonomic and central nervous system function.",
                "Suspect": "Diminished accelerations may reflect fetal quiescence or mild compromise.",
                "Pathological": "Absent accelerations suggest impaired fetal responsiveness or possible hypoxia."
            },
            "fetal_movement": {
                "Normal": "Active movements reflect a healthy fetal neuro-muscular status.",
                "Suspect": "Reduced movements may suggest transient rest phases or mild compromise.",
                "Pathological": "Markedly diminished movements indicate potential hypoxia or neurologic dysfunction."
            },
            "uterine_contractions": {
                "Normal": "Regular moderate contractions support normal uteroplacental function",
                "Suspect": "Increased frequency may transiently reduce fetal oxygenation and requires monitoring.",
                "Pathological": "Excessive or prolonged contractions may impair placental perfusion and induce fetal distress."
            },
            "light_decelerations": {
                "Normal": "Isolated mild decelerations are typically benign and related to fetal movement.",
                "Suspect": "Frequent episodes may indicate transient umbilical cord compression.",
                "Pathological": "Recurrent or prolonged decelerations may reflect compromised fetal oxygenation."
            },
            "severe_decelerations": {
                "Normal": "No severe decelerations or only occasional occurrences suggest stable fetal well-being.",
                "Suspect": "Occasional severe decelerations may be present, warranting further investigation to assess fetal status.",
                "Pathological": "Frequent or prolonged decelerations are indicative of significant fetal distress, requiring immediate intervention."

            },
            "prolongued_decelerations": {
                "Normal": "Absent under normal circumstances, indicating no fetal distress.",
                "Suspect": "Occasional prolonged decelerations may occur, requiring further evaluation to determine fetal well-being.",
                "Pathological": "Persistent prolonged decelerations are indicative of fetal compromise and may necessitate urgent medical intervention."
            },
            "abnormal_short_term_variability": {
                "Normal": "Normal variability in heart rate indicates a healthy autonomic system.",
                "Suspect": "Mildly reduced variability may indicate fetal sleep or early signs of distress.",
                "Pathological": "Significantly reduced variability signals potential fetal compromise and requires further assessment."
            },
            "mean_value_of_short_term_variability": {
                "Normal": "Falls within a typical range, indicating good fetal reactivity.",
                "Suspect": "Mildly reduced variability requires observation to assess fetal condition.",
                "Pathological": "Extremely low variability may suggest fetal distress or central nervous system dysfunction."
            },
            "percentage_of_time_with_abnormal_long_term_variability": {
                "Normal": "Minimal abnormal variability is expected in a healthy fetus.",
                "Suspect": "Increased abnormal variability suggests an irregular fetal response to stimuli, requiring monitoring.",
                "Pathological": "High percentages of abnormal variability indicate significant instability in fetal heart rate control."
            },
            "mean_value_of_long_term_variability": {
                "Normal": "Fluctuations within the expected range reflect a healthy fetal nervous system.",
                "Suspect": "Mildly abnormal values may suggest minor fetal stress, warranting monitoring.",
                "Pathological": "Severely reduced variability indicates significant fetal distress, requiring urgent attention."
            },
            "histogram_width": {
                "Normal": "A well-distributed heart rate pattern suggests stable fetal health.",
                "Suspect": "A narrow or excessively broad range may indicate altered fetal heart rate control, requiring further evaluation.",
                "Pathological": "Extreme values suggest severe fetal distress or arrhythmia, needing immediate intervention."

            },
            "histogram_min": {
                "Normal": "The lowest recorded fetal heart rate is within a safe range, indicating no concern.",
                "Suspect": "Slightly lower values may require further observation to assess fetal status.",
                "Pathological": "Very low values may indicate fetal bradycardia, signaling potential distress and the need for urgent care."
            },
            "histogram_max": {
                "Normal": "Highest recorded fetal heart rate within the expected range, indicating normal fetal well-being.",
                "Suspect": "Slight elevations may reflect temporary fetal stress, warranting monitoring.",
                "Pathological": "Extremely high values suggest fetal tachycardia, requiring immediate intervention."

            },
            "histogram_number_of_peaks": {
                "Normal": "Normal variation in heart rate, indicating healthy autonomic function.",
                "Suspect": "Fewer peaks may indicate reduced heart rate variability, suggesting early signs of distress.",
                "Pathological": "A flat histogram suggests a lack of variability, indicating potential fetal compromise."

            },
            "histogram_number_of_zeroes": {
                "Normal": "Few or no zeroes observed, indicating normal heart rate activity.",
                "Suspect": "Increased zeroes may indicate erratic heart rate, warranting further assessment.",
                "Pathological": "High zero counts may suggest recording errors or a serious condition such as cardiac arrest."
            },
            "histogram_mode": {
                "Normal": "The most frequent heart rate is within a healthy range, indicating normal fetal function.",
                "Suspect": "Shifts in mode suggest altered heart rate regulation, requiring further investigation.",
                "Pathological": "Mode outside the normal fetal heart rate range signals fetal compromise and the need for immediate evaluation."
            },
            "histogram_mean": {
                "Normal": "The mean value reflects a stable and typical heart rate distributio.",
                "Suspect": "Slight shifts may indicate transient fetal distress, requiring observation.",
                "Pathological": "A significant deviation suggests irregular heart rate regulation, signaling possible fetal compromise."
            },
            "histogram_median": {
                "Normal": "The median aligns closely with the mean, indicating a normal heart rate pattern.",
                "Suspect": "Differences between the median and mean may suggest irregularities, warranting further evaluation.",
                "Pathological": "A large gap indicates asymmetric heart rate patterns, possibly due to fetal distress."

            },
            "histogram_variance": {
                "Normal": "A balanced variance suggests stable heart rate variation.",
                "Suspect": "Increased variance may indicate instability in the heart rate pattern, requiring monitoring.",
                "Pathological": "Extremely high or low variance signals potential fetal distress, needing urgent attention."

            },
            "histogram_tendency": {
                "Normal": "A stable heart rate tendency with mild fluctuations indicates healthy fetal status.",
                "Suspect": "Increasing or decreasing tendencies may suggest temporary stress and require observation.",
                "Pathological": "A strong tendency toward bradycardia or tachycardia is concerning and may indicate fetal compromise."
            }
        }

        

        explanation_text = []
        for feature, shap_value in top_3_features:
            impact_direction = "increasing" if shap_value > 0 else "decreasing"
            explanation = feature_explanations.get(feature, {}).get(prediction_label, "No explanation available.")
            explanation_text.append(f" * {feature} has a SHAP value of {shap_value:.4f}, {impact_direction} the probability of {prediction_label}.\n  {explanation}")

        
        overall_interpretation = f"Fetal Health Condition is {prediction_label}.\n"
        overall_interpretation += f"This decision was mostly influenced by {top_3_features[0][0]}, {top_3_features[1][0]}, and {top_3_features[2][0]}."

        
        lab_reports_collection.update_one(
            {'_id': ObjectId(report_id)},
            {'$set': {
                'prediction': prediction_label,
                'probabilities': prediction_probabilities,
                'top_features': top_3_features
            }}
        )

        return render_template("predict_result.html",
                               report_id=report_id,
                               username=report.get('patient_username', 'Unknown'),
                               phone=report.get('patient_phone', 'Unknown'),
                               lab_assistant=report.get('lab_assistant_username', 'Unknown'),
                               prediction=prediction_label,
                               probabilities=prediction_probabilities,
                               top_features=top_3_features,
                               shap_all_path=shap_all_path,
                               shap_top3_path=shap_top3_path,
                               explanation_text=explanation_text,
                               overall_interpretation=overall_interpretation)

    except Exception as e:
        return render_template("error.html", error_message=str(e))



@app.route('/delete_report_doctor/<report_id>', methods=['POST'])
def delete_report_doctor(report_id):
    if "username" not in session or session["role"] != "doctor":
        return redirect(url_for("login"))

    lab_reports_collection = mongo.db.lab_reports

    
    lab_reports_collection.update_one(
        {'_id': ObjectId(report_id)},
        {'$set': {'deleted_by_doctor': True}}
    )


    return redirect(url_for("doctor_dashboard"))


@app.route('/add_doctor_message/<report_id>', methods=['POST'])
def add_doctor_message(report_id):
    try:
        message = request.form['doctor_message']
        lab_reports_collection = mongo.db.lab_reports  
        lab_reports_collection.update_one({'_id': ObjectId(report_id)}, {'$set': {'doctor_message': message}})

        return redirect(url_for('doctor_dashboard'))
    except Exception as e:
        return f"Error in adding doctor's message: {e}"






@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))




if __name__ == "__main__":
    app.run(debug=True)
