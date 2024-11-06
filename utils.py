import sys
from django.core.management import execute_from_command_line
import pandas as pd
import nltk
from gensim.models import Word2Vec
import pickle
from trainedmodel import vectorise_comments, clean_comments
import gensim

def run_migrations():
    execute_from_command_line(sys.argv)

if __name__ == "__main__":
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    run_migrations()

def load_models(model_path, word2vec_path):
    with open(model_path, 'rb') as model_file, open(word2vec_path, 'rb') as word2vec_file:
        rfg_model = pickle.load(model_file)
        word2vec_model = pickle.load(word2vec_file)
    return rfg_model, word2vec_model

def predict_sentiment(comment, rfg_model, word2vec_model):
        # go through our pre-processing steps of cleaning and vectorising the comment
        comments = clean_comments([comment])
        X = vectorise_comments(comments, word2vec_model)
        predicted_sentiment = rfg_model.predict(X)[0] # then use the trained model to predict a sentiment
        sentiment_scores.append(predicted_sentiment) # add prediction to the list of sentiment scores



df = pd.read_csv('ModuleOpinions.csv', header=None) # read student feedback data from CSV file
module_titles = df.iloc[0] # get module titles from first row
module_comments = df.iloc[1] # get comments from second row
module_comments_dict = {} # dictionary to store comments for each module

for title, comments in zip(module_titles, module_comments): # loop through each pair of a module title and its corresponding comments
    split_comments = comments.split('\n')
    cleaned_comments = []
    # remove any whitespace from each comment and filter out any empty comments
    for comment in split_comments:
        stripped_comment = comment.strip()
        if stripped_comment:  # make sure comment isn't empty after stripping before adding to the list
            cleaned_comments.append(stripped_comment)
    module_comments_dict[title] = cleaned_comments # add the list of cleaned comments to the corresponding module title in the dictionary

algocomp_comments = module_comments_dict.get('Algorithms and Complexity', [])
advancedfp_comments = module_comments_dict.get('Advanced Functional Programming', [])
advancednetworking_comments = module_comments_dict.get('Advanced Networking', [])
cvi_comments = module_comments_dict.get('Computer Aided Verification', [])
cav_comments = module_comments_dict.get('Computer Vision and Imaging', [])
dds_comments = module_comments_dict.get('Dependable and Distributed Systems', [])
evocomp_comments = module_comments_dict.get('Evolutionary Computation', [])
hci_comments = module_comments_dict.get('Human Computer Interaction', [])
iis_comments = module_comments_dict.get('Intelligent Interactive Systems', [])
robotics_comments = module_comments_dict.get('Intelligent Robotics', [])
ML_comments = module_comments_dict.get('Machine Learning', [])
IDA_comments = module_comments_dict.get('Intelligent Data Analysis', [])
mobile_comments = module_comments_dict.get('Mobile and Ubiquitous Computing', [])
NLP_comments = module_comments_dict.get('Natural Language Processing', [])
neural_comments = module_comments_dict.get('Neural Computation', [])
PLPDI_comments = module_comments_dict.get('Programming Language Principles, Design, and Implementation', [])
security_comments = module_comments_dict.get('Security of Real World Systems', [])
securesoftware_comments = module_comments_dict.get('Secure Software and Hardware Systems', [])
teachingcs_comments = module_comments_dict.get('Teaching Computer Science in Schools', [])

sentiment_scores = [] # list to store sentiment scores for comments for a specific module
rfg_model, word2vec_model = load_models('C:/Users/drago/IdeaProjects/FYP/random_forest_regressor.pkl', 'C:/Users/drago/IdeaProjects/FYP/word2vec_model.pkl')


def calculate_algo_comp_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    math_preference = 'Maths' in preferences
    complexity_preference = 'Complexity' in preferences
    logic_preference = 'Logic' in preferences
    algorithms_preference = 'Algorithms' in preferences
    software_engineer_aspiration = 'Software Engineer' in career_aspirations
    data_scientist_aspiration = 'Data Scientist' in career_aspirations
    machine_learning_aspiration = 'Machine Learning' in career_aspirations
    quantitative_analyst_aspiration = 'Quantitative Analyst' in career_aspirations

    for comment in algocomp_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)

    score = average_sentiment_score
    if math_preference:
        score *= 1.3
    if logic_preference:
        score *= 1.3
    if algorithms_preference:
        score *= 1.3
    if complexity_preference:
        score *= 2.2
    if software_engineer_aspiration:
        score *= 1.3
    if data_scientist_aspiration:
        score *= 2.2
    if machine_learning_aspiration:
        score *= 2.2
    if quantitative_analyst_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_advanced_fp_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    math_preference = 'Maths' in preferences
    fp_preference = 'Functional Programming' in preferences
    logic_preference = 'Logic' in preferences
    software_engineer_aspiration = 'Software Engineer' in career_aspirations

    for comment in advancedfp_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if math_preference:
        score *= 1.3
    if fp_preference:
        score *= 2.2
    if software_engineer_aspiration:
        score *= 1.3
    if logic_preference:
        score *= 1.3

    return round(score, 3)

def calculate_advanced_networking_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    networking_preference = 'Networking' in preferences
    networking_aspiration = 'Networking' in career_aspirations

    for comment in advancednetworking_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if networking_preference:
        score *= 2.2
    if networking_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_cav_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    maths_preference = 'Maths' in preferences
    formal_methods_preference = 'Formal Methods' in preferences
    logic_preference = 'Logic' in preferences
    verification_engineer_aspiration = 'Verification Engineer' in career_aspirations
    software_engineer_aspiration = 'Software Engineer' in career_aspirations

    for comment in cav_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if formal_methods_preference:
        score *= 2.2
    if logic_preference:
        score *= 1.3
    if maths_preference:
        score *= 1.3
    if software_engineer_aspiration:
        score *= 1.3
    if verification_engineer_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_cvi_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    maths_preference = 'Maths' in preferences
    algorithm_preference = 'Algorithms' in preferences
    computer_vision_preference = 'Computer Vision' in preferences
    machine_learning_preference = 'Machine Learning' in preferences
    programming_preference = 'MATLAB' in preferences
    computer_vision_aspiration = 'Computer Vision Engineer' in career_aspirations
    software_engineer_aspiration = 'Software Engineer' in career_aspirations
    research_aspiration = 'Research' in career_aspirations

    for comment in cvi_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if algorithm_preference:
        score *= 1.3
    if computer_vision_preference:
        score *= 2.2
    if programming_preference:
        score *= 1.3
    if maths_preference:
        score *= 1.3
    if machine_learning_preference:
        score *= 1.3
    if software_engineer_aspiration:
        score *= 1.3
    if computer_vision_aspiration:
        score *= 2.2
    if research_aspiration:
        score *= 1.3

    return round(score, 3)

def calculate_dds_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    distributed_systems_preference = 'Distributed Systems' in preferences
    maths_preference = 'Maths' in preferences
    software_developer_aspiration = 'Software Developer' in career_aspirations
    software_engineer_aspiration = 'Software Engineer' in career_aspirations

    for comment in dds_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if distributed_systems_preference:
        score *= 2.2
    if maths_preference:
        score *= 1.3
    if software_engineer_aspiration:
        score *= 2.2
    if software_developer_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_evocomp_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    maths_preference = 'Maths' in preferences
    evolutionary_algorithms_preference = 'Evolutionary Algorithms' in preferences
    machine_learning_preference = 'Machine Learning' in preferences
    programming_preference = 'MATLAB' in preferences
    research_aspiration = 'Research' in career_aspirations
    data_scientist_aspiration = 'Data Scientist' in career_aspirations

    for comment in evocomp_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if evolutionary_algorithms_preference:
        score *= 2.2
    if programming_preference:
        score *= 1.3
    if maths_preference:
        score *= 1.3
    if machine_learning_preference:
        score *= 1.3
    if research_aspiration:
        score *= 2.2
    if data_scientist_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_hci_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    psychology_preference = 'Psychology' in preferences
    design_problem_preference = 'Design Problems' in preferences
    ux_designer_aspiration = 'UX Designer' in career_aspirations
    research_aspiration = 'Research' in career_aspirations

    for comment in hci_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if psychology_preference:
        score *= 2.2
    if design_problem_preference:
        score *= 1.3
    if ux_designer_aspiration:
        score *= 2.2
    if research_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_iis_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    psychology_preference = 'Psychology' in preferences
    design_problem_preference = 'Design Problems' in preferences
    ux_designer_aspiration = 'UX Designer' in career_aspirations

    for comment in iis_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if psychology_preference:
        score *= 2.2
    if design_problem_preference:
        score *= 1.3
    if ux_designer_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_robotics_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    robotics_preference = 'Robotics' in preferences
    programming_preference = 'Python' in preferences
    maths_preference = 'Maths' in preferences
    robotics_engineer_career_aspirations = 'Robotics Engineer' in career_aspirations
    ai_ml_engineer_career_aspirations = 'AI/ML Engineer' in career_aspirations
    research_career_aspirations = 'Research' in career_aspirations

    for comment in robotics_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if robotics_preference:
        score *= 2.2
    if programming_preference:
        score *= 1.3
    if maths_preference:
        score *= 1.3
    if robotics_engineer_career_aspirations:
        score *= 2.2
    if ai_ml_engineer_career_aspirations:
        score *= 2.2
    if research_career_aspirations:
        score *= 1.3

    return round(score, 3)

def calculate_ML_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    machine_learning_preference = 'Machine Learning' in preferences
    maths_preference = 'Maths' in preferences
    machine_learning_aspirations = 'Machine Learning' in career_aspirations
    software_engineer_aspirations = 'Software Engineer' in career_aspirations

    for comment in ML_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if machine_learning_preference:
        score *= 2.2
    if maths_preference:
        score *= 1.3
    if machine_learning_aspirations:
        score *= 2.2
    if software_engineer_aspirations:
        score *= 1.3

    return round(score, 3)

def calculate_ida_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    programming_preference = 'MATLAB' in preferences
    programming_preference2 = 'Python' in preferences
    data_analysis_preference = 'Data Analysis' in preferences
    data_analyst_aspiration = 'Data Analyst' in career_aspirations

    for comment in IDA_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if programming_preference:
        score *= 1.6
    if programming_preference2:
        score *= 1.3
    if data_analysis_preference:
        score *= 2.2
    if data_analyst_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_mobile_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    mobile_preference = 'Mobile' in preferences
    design_problem_preference = 'Design Problems' in preferences
    mobile_app_developer = 'Mobile App Developer' in career_aspirations

    for comment in mobile_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if mobile_preference:
        score *= 2.2
    if design_problem_preference:
        score *= 1.3
    if mobile_app_developer:
        score *= 2.2

    return round(score, 3)


def calculate_NLP_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    nlp_preference = 'NLP' in preferences
    ai_preference = 'Artificial Intelligence' in preferences
    maths_preference = 'Maths' in preferences
    ai_ml_career_aspiration = 'AI/ML Engineer' in career_aspirations
    data_science_aspiration = 'Data Scientist' in career_aspirations
    research_aspiration = 'Research' in career_aspirations

    for comment in NLP_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if nlp_preference:
        score *= 2.2
    if ai_preference:
        score *= 1.3
    if maths_preference:
        score *= 1.3
    if ai_ml_career_aspiration:
        score *= 2.2
    if data_science_aspiration:
        score *= 2.2
    if research_aspiration:
        score *= 1.3

    return round(score, 3)

def calculate_neural_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    maths_preference = 'Maths' in preferences
    ml_preference = 'Machine Learning' in preferences
    neural_networks_preference = 'Artificial Neural Networks' in preferences
    ai_ml_career_aspiration = 'AI/ML Engineer' in career_aspirations
    data_science_aspiration = 'Data Scientist' in career_aspirations
    research_aspiration = 'Research' in career_aspirations

    for comment in neural_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if neural_networks_preference:
        score *= 2.2
    if ml_preference:
        score *= 1.3
    if maths_preference:
        score *= 1.3
    if ai_ml_career_aspiration:
        score *= 2.2
    if data_science_aspiration:
        score *= 2.2
    if research_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_PLPDI_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    programming_preference = 'Programming' in preferences
    software_engineer_aspiration = 'Software Engineer' in career_aspirations
    compiler_engineer_aspiration = 'Compiler Engineer' in career_aspirations

    for comment in PLPDI_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if programming_preference:
        score *= 2.2
    if software_engineer_aspiration:
        score *= 1.3
    if compiler_engineer_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_security_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    security_preference = 'Security' in preferences
    cybersecurity_aspiration = 'Cybersecurity' in career_aspirations

    for comment in security_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if security_preference:
        score *= 2.2
    if cybersecurity_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_securesoftware_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    security_preference = 'Security' in preferences
    cybersecurity_aspiration = 'Cybersecurity' in career_aspirations
    security_engineer_aspiration = 'Security Engineer' in career_aspirations

    for comment in securesoftware_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if security_preference:
        score *= 2.2
    if cybersecurity_aspiration:
        score *= 2.2
    if security_engineer_aspiration:
        score *= 2.2

    return round(score, 3)

def calculate_teachingcs_score(user):
    sentiment_scores.clear() # make sure list is empty before calculating sentiment scores for a specific module
    # Retrieve user's preferences and career aspirations from the database
    preferences = user.preferences.split(', ') if user.preferences else []
    career_aspirations = user.career_aspirations.split(', ') if user.career_aspirations else []

    # Check if these specific preferences and career aspirations exist
    teaching_preference = 'Teaching' in preferences
    teaching_aspiration = 'CS Teacher' in career_aspirations

    for comment in teachingcs_comments:
        predict_sentiment(comment, rfg_model, word2vec_model)

    if sentiment_scores:
        average_hybrid_sentiment = sum(sentiment_scores) / len(sentiment_scores)

    score = average_hybrid_sentiment
    if teaching_preference:
        score *= 2.2
    if teaching_aspiration:
        score *= 2.2

    return round(score, 3)
