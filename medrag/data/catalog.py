"""Clinical vocabulary used to synthesise realistic radiology studies and reports."""

MODALITIES = ["CT", "MRI", "XR", "US", "MG", "PET"]

MODALITY_NAMES = {
    "CT": "CT", "MRI": "MRI", "XR": "X-ray", "US": "Ultrasound", "MG": "Mammography", "PET": "PET-CT",
}

# Synonyms seen in dirty source data and in user questions -> canonical modality code.
MODALITY_SYNONYMS = {
    "ct": "CT", "ct scan": "CT", "cat scan": "CT", "computed tomography": "CT",
    "mri": "MRI", "mr": "MRI", "magnetic resonance": "MRI",
    "xr": "XR", "xray": "XR", "x-ray": "XR", "x ray": "XR", "radiograph": "XR", "cxr": "XR",
    "us": "US", "usg": "US", "ultrasound": "US", "sonography": "US", "ultrasonography": "US",
    "mg": "MG", "mammo": "MG", "mammogram": "MG", "mammography": "MG",
    "pet": "PET", "pet-ct": "PET", "pet ct": "PET", "pet scan": "PET",
}

PRIORITIES = ["routine", "urgent", "stat"]
PRIORITY_SYNONYMS = {
    "routine": "routine", "low": "routine", "normal": "routine", "medium": "routine", "elective": "routine",
    "urgent": "urgent", "high": "urgent", "priority": "urgent",
    "stat": "stat", "critical": "stat", "emergency": "stat", "immediate": "stat",
}

REPORT_STATUSES = ["final", "preliminary", "amended"]
STATUS_SYNONYMS = {
    "final": "final", "approved": "final", "signed": "final", "finalized": "final",
    "preliminary": "preliminary", "draft": "preliminary", "prelim": "preliminary", "pending": "preliminary",
    "amended": "amended", "addendum": "amended", "corrected": "amended",
}

GENDER_SYNONYMS = {
    "m": "male", "male": "male", "man": "male",
    "f": "female", "female": "female", "woman": "female",
    "o": "other", "other": "other", "non-binary": "other",
}

BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

CHRONIC_CONDITIONS = [
    "Type 2 diabetes mellitus", "Hypertension", "Chronic obstructive pulmonary disease", "Asthma",
    "Coronary artery disease", "Chronic kidney disease", "Hypothyroidism", "Osteoarthritis",
    "Rheumatoid arthritis", "Hyperlipidemia", "Atrial fibrillation", "Heart failure",
    "Non-alcoholic fatty liver disease", "Migraine", "Epilepsy", "Breast cancer (treated)",
    "Lung cancer (on surveillance)", "Colorectal cancer (treated)", "Tuberculosis (treated)",
    "Obesity", "Gout", "Osteoporosis", "Parkinson disease", "Prostate enlargement (BPH)",
]

# (modality, body part) -> technique line
TECHNIQUES = {
    ("CT", "chest"): "Contrast-enhanced CT of the chest with axial 1.25 mm reconstructions.",
    ("CT", "abdomen"): "CT abdomen and pelvis with oral and intravenous contrast, portal venous phase.",
    ("CT", "head"): "Non-contrast CT of the head with axial and coronal reformats.",
    ("CT", "spine"): "Non-contrast CT of the lumbar spine with sagittal and coronal reformats.",
    ("MRI", "brain"): "Multiplanar multisequence MRI brain with and without gadolinium (T1, T2, FLAIR, DWI, SWI).",
    ("MRI", "spine"): "MRI lumbar spine, sagittal T1/T2 and axial T2 sequences.",
    ("MRI", "knee"): "MRI right knee, proton density fat-saturated sequences in three planes.",
    ("MRI", "abdomen"): "MRI abdomen with hepatobiliary contrast and diffusion-weighted imaging.",
    ("XR", "chest"): "PA and lateral views of the chest.",
    ("XR", "knee"): "AP and lateral weight-bearing views of the knee.",
    ("XR", "spine"): "AP and lateral views of the lumbar spine.",
    ("XR", "hand"): "PA, oblique and lateral views of the hand.",
    ("US", "abdomen"): "Grey-scale and colour Doppler ultrasound of the abdomen.",
    ("US", "thyroid"): "High-resolution ultrasound of the thyroid gland and neck.",
    ("US", "pelvis"): "Transabdominal and transvaginal pelvic ultrasound.",
    ("MG", "breast"): "Bilateral digital mammography with tomosynthesis, CC and MLO views.",
    ("PET", "whole body"): "FDG PET-CT from skull base to mid-thigh, 60 minutes after 10 mCi FDG.",
}

NORMAL_FINDINGS = {
    "chest": "The lungs are clear without consolidation, effusion or pneumothorax. Heart size is normal. Mediastinal contours are unremarkable.",
    "abdomen": "The liver, spleen, pancreas, adrenal glands and kidneys are unremarkable. No free fluid or lymphadenopathy.",
    "head": "No intracranial haemorrhage, mass effect or midline shift. Ventricles and sulci are normal for age.",
    "brain": "Normal grey-white matter differentiation. No restricted diffusion, mass or abnormal enhancement.",
    "spine": "Vertebral body heights and alignment are maintained. No significant canal or foraminal stenosis.",
    "knee": "Menisci and cruciate ligaments are intact. No joint effusion. Articular cartilage is preserved.",
    "hand": "No fracture or dislocation. Joint spaces are preserved. Bone mineralisation is normal.",
    "thyroid": "Both thyroid lobes are normal in size and echotexture. No focal nodule.",
    "pelvis": "Uterus and ovaries are normal in size and appearance. No adnexal mass or free fluid.",
    "breast": "Scattered fibroglandular densities. No suspicious mass, calcification or architectural distortion.",
    "whole body": "No abnormal FDG-avid lesion. Physiological uptake in brain, myocardium and urinary tract.",
}

# Abnormal findings: label, body parts it applies to, findings text, impression, recommendation, severity (1-3).
FINDINGS = [
    ("pneumonia", ["chest"], "Patchy airspace consolidation in the {side} lower lobe with air bronchograms.",
     "Findings consistent with {side} lower lobe pneumonia.", "Clinical correlation and follow-up radiograph in 6 weeks after treatment.", 2),
    ("pulmonary nodule", ["chest"], "A {size} mm solid nodule in the {side} upper lobe, smooth margins.",
     "Indeterminate {size} mm pulmonary nodule.", "Follow-up CT chest in 6-12 months per Fleischner guidelines.", 2),
    ("lung mass", ["chest"], "A {big} cm spiculated mass in the {side} upper lobe with ipsilateral hilar lymphadenopathy.",
     "Spiculated lung mass highly suspicious for primary lung malignancy.", "Urgent referral to thoracic oncology and tissue sampling.", 3),
    ("pleural effusion", ["chest"], "Moderate {side} pleural effusion with adjacent compressive atelectasis.",
     "Moderate {side} pleural effusion.", "Consider diagnostic thoracentesis if clinically indicated.", 2),
    ("pneumothorax", ["chest"], "A {side} apical pneumothorax measuring approximately {size} mm.",
     "{side} pneumothorax.", "Immediate clinical review; consider chest drain.", 3),
    ("cardiomegaly", ["chest"], "Cardiothoracic ratio is increased at 0.58. Mild pulmonary venous congestion.",
     "Cardiomegaly with mild pulmonary venous congestion.", "Echocardiography recommended.", 2),
    ("emphysema", ["chest"], "Centrilobular emphysematous changes predominantly in the upper lobes.",
     "Centrilobular emphysema.", "Correlate with pulmonary function tests.", 1),
    ("pulmonary embolism", ["chest"], "Filling defects within the {side} lower lobe segmental pulmonary arteries.",
     "Acute {side} segmental pulmonary embolism.", "Immediate anticoagulation per clinical protocol.", 3),
    ("hepatic steatosis", ["abdomen"], "Diffusely decreased hepatic attenuation/increased echogenicity in keeping with fatty infiltration.",
     "Hepatic steatosis.", "Lifestyle modification and liver function monitoring.", 1),
    ("liver lesion", ["abdomen"], "A {big} cm arterially enhancing lesion in segment {seg} of the liver with washout.",
     "Arterially enhancing liver lesion with washout, LI-RADS 5, suspicious for hepatocellular carcinoma.", "Multidisciplinary hepatobiliary review.", 3),
    ("gallstones", ["abdomen"], "Multiple mobile echogenic calculi in the gallbladder with posterior acoustic shadowing. No wall thickening.",
     "Cholelithiasis without cholecystitis.", "Surgical opinion if symptomatic.", 1),
    ("renal calculus", ["abdomen"], "A {size} mm calculus in the {side} renal pelvis with mild hydronephrosis.",
     "{side} renal calculus with mild hydronephrosis.", "Urology referral.", 2),
    ("appendicitis", ["abdomen"], "Dilated, non-compressible appendix measuring 11 mm with periappendiceal fat stranding.",
     "Acute appendicitis.", "Urgent surgical review.", 3),
    ("splenomegaly", ["abdomen"], "The spleen is enlarged measuring 15.8 cm in craniocaudal length.",
     "Splenomegaly.", "Correlate with haematological parameters.", 1),
    ("intracranial haemorrhage", ["head", "brain"], "Acute intraparenchymal haemorrhage in the {side} basal ganglia measuring {big} cm with surrounding oedema.",
     "Acute {side} basal ganglia haemorrhage.", "Immediate neurosurgical consultation.", 3),
    ("ischaemic stroke", ["head", "brain"], "Restricted diffusion / hypodensity in the {side} middle cerebral artery territory.",
     "Acute {side} MCA territory infarct.", "Stroke team activation.", 3),
    ("brain tumour", ["brain"], "A {big} cm heterogeneously enhancing mass in the {side} frontal lobe with surrounding vasogenic oedema.",
     "Enhancing {side} frontal mass, likely high-grade glioma.", "Neurosurgical and neuro-oncology referral.", 3),
    ("small vessel disease", ["head", "brain"], "Scattered periventricular and subcortical white matter hyperintensities.",
     "Chronic small vessel ischaemic change, Fazekas grade 2.", "Vascular risk factor optimisation.", 1),
    ("disc herniation", ["spine"], "Posterior disc extrusion at L{lvl}-L{lvl1} compressing the traversing {side} nerve root.",
     "L{lvl}-L{lvl1} disc extrusion with {side} nerve root compression.", "Spine clinic referral; consider MRI-guided management.", 2),
    ("compression fracture", ["spine"], "Wedge compression of the L{lvl} vertebral body with approximately 30% height loss.",
     "L{lvl} compression fracture, likely osteoporotic.", "DEXA scan and orthopaedic review.", 2),
    ("degenerative spondylosis", ["spine"], "Multilevel disc desiccation, osteophytes and facet arthropathy.",
     "Multilevel degenerative spondylosis.", "Conservative management.", 1),
    ("meniscal tear", ["knee"], "Oblique tear of the posterior horn of the medial meniscus extending to the inferior articular surface.",
     "Medial meniscus posterior horn tear.", "Orthopaedic review.", 2),
    ("ACL tear", ["knee"], "Discontinuity of the anterior cruciate ligament fibres with bone marrow oedema of the lateral femoral condyle.",
     "Complete ACL tear.", "Sports orthopaedic referral.", 2),
    ("osteoarthritis", ["knee", "hand"], "Joint space narrowing, subchondral sclerosis and marginal osteophytes.",
     "Moderate osteoarthritis.", "Conservative management and physiotherapy.", 1),
    ("fracture", ["hand", "knee"], "Transverse fracture of the {bone} with minimal displacement.",
     "Acute minimally displaced {bone} fracture.", "Orthopaedic review and immobilisation.", 2),
    ("thyroid nodule", ["thyroid"], "A {size} mm solid hypoechoic nodule in the {side} lobe with microcalcifications.",
     "{side} thyroid nodule, TI-RADS 5.", "Fine-needle aspiration recommended.", 2),
    ("ovarian cyst", ["pelvis"], "A {big} cm simple-appearing cyst in the {side} ovary.",
     "Simple {side} ovarian cyst.", "Follow-up ultrasound in 8-12 weeks.", 1),
    ("uterine fibroid", ["pelvis"], "An intramural fibroid measuring {big} cm in the uterine body.",
     "Uterine fibroid.", "Gynaecology review if symptomatic.", 1),
    ("breast mass", ["breast"], "An irregular {size} mm spiculated mass at the {clock} o'clock position of the {side} breast.",
     "Suspicious {side} breast mass, BI-RADS 5.", "Ultrasound-guided core biopsy.", 3),
    ("breast calcifications", ["breast"], "Grouped fine pleomorphic calcifications in the {side} upper outer quadrant.",
     "Suspicious {side} breast calcifications, BI-RADS 4.", "Stereotactic biopsy.", 2),
    ("metastatic disease", ["whole body"], "Multiple FDG-avid lesions in the liver and skeleton, SUVmax {suv}.",
     "Widespread FDG-avid metastatic disease.", "Oncology review.", 3),
    ("lymphoma", ["whole body"], "FDG-avid cervical, mediastinal and retroperitoneal lymphadenopathy, SUVmax {suv}.",
     "FDG-avid lymphadenopathy above and below the diaphragm, consistent with lymphoma.", "Haematology-oncology referral and biopsy.", 3),
]

# Findings that only some modalities can show (anything not listed: every modality imaging that body part).
FINDING_MODALITIES = {
    "pulmonary embolism": {"CT"}, "liver lesion": {"CT", "MRI"}, "appendicitis": {"CT", "US"},
    "emphysema": {"CT", "XR"}, "pulmonary nodule": {"CT", "XR"}, "intracranial haemorrhage": {"CT", "MRI"},
    "ischaemic stroke": {"CT", "MRI"}, "meniscal tear": {"MRI"}, "ACL tear": {"MRI"}, "disc herniation": {"MRI", "CT"},
    "hepatic steatosis": {"CT", "US", "MRI"}, "gallstones": {"US", "CT"}, "splenomegaly": {"US", "CT", "MRI"},
}

# Which body parts each modality images, with relative frequencies.
MODALITY_BODY_PARTS = {
    "CT": {"chest": 4, "abdomen": 4, "head": 3, "spine": 1},
    "MRI": {"brain": 4, "spine": 3, "knee": 2, "abdomen": 1},
    "XR": {"chest": 6, "knee": 2, "spine": 1, "hand": 1},
    "US": {"abdomen": 5, "thyroid": 2, "pelvis": 2},
    "MG": {"breast": 1},
    "PET": {"whole body": 1},
}
MODALITY_WEIGHTS = {"CT": 25, "MRI": 15, "XR": 35, "US": 18, "MG": 4, "PET": 3}

INDICATIONS = {
    "chest": ["Cough and fever", "Shortness of breath", "Chest pain", "Haemoptysis", "Follow-up of lung nodule", "Pre-operative assessment"],
    "abdomen": ["Right upper quadrant pain", "Abdominal pain", "Deranged liver function tests", "Haematuria", "Weight loss", "Follow-up of liver lesion"],
    "head": ["Head injury", "Sudden onset headache", "Acute confusion", "Syncope"],
    "brain": ["Seizure", "Chronic headache", "Focal neurological deficit", "Memory loss", "Follow-up of brain lesion"],
    "spine": ["Low back pain", "Sciatica", "Back pain after fall", "Leg numbness"],
    "knee": ["Knee pain after sports injury", "Chronic knee pain", "Knee swelling"],
    "hand": ["Fall on outstretched hand", "Joint pain and stiffness"],
    "thyroid": ["Neck swelling", "Abnormal thyroid function", "Palpable neck lump"],
    "pelvis": ["Pelvic pain", "Abnormal uterine bleeding", "Infertility work-up"],
    "breast": ["Screening mammogram", "Palpable breast lump", "Nipple discharge"],
    "whole body": ["Cancer staging", "Treatment response assessment", "Unexplained weight loss"],
}

AI_MODELS = {
    "CT": "DeepCT-Seg v4.2", "MRI": "NeuroMR-Net v2.1", "XR": "ChestXR-Vision v3.0",
    "US": "SonoAssist v1.8", "MG": "MammoScreen-AI v2.5", "PET": "PET-Lesion v1.3",
}

BONES = ["distal radius", "fifth metacarpal", "scaphoid", "proximal phalanx", "patella", "tibial plateau"]
PROGRESSION = {
    "stable": "Unchanged compared with the prior study of {prior}.",
    "improved": "Interval improvement compared with the prior study of {prior}.",
    "worse": "Interval progression compared with the prior study of {prior}.",
}
