from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.contrib.sites.shortcuts import get_current_site
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes
from django.contrib.auth.tokens import default_token_generator
from FYP.forms import registerform
from OptiMods.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.backends import ModelBackend
from FYP.forms import PreferenceForm, CareerForm
from utils import calculate_algo_comp_score
from utils import calculate_cav_score
from utils import calculate_cvi_score
from utils import calculate_dds_score
from utils import calculate_hci_score
from utils import calculate_advanced_fp_score
from utils import calculate_advanced_networking_score
from utils import calculate_evocomp_score
from utils import calculate_ida_score
from utils import calculate_iis_score
from utils import calculate_ML_score
from utils import calculate_mobile_score
from utils import calculate_neural_score
from utils import calculate_NLP_score
from utils import calculate_PLPDI_score
from utils import calculate_robotics_score
from utils import calculate_security_score
from utils import calculate_securesoftware_score
from utils import calculate_teachingcs_score
from FYP.forms import FeedbackForm
from utils import algocomp_comments

def index(request):
    return HttpResponse("Hello, this is your Django app view.")

def my_view(request):
    # Your view logic goes here
    context = {
        'variable_name': 'some_value',
        # Add more context variables as needed
    }
    return render(request, 'login.html', context)

def register(request):
    # check if form was submitted and create form instance populated with the data
    if request.method == 'POST':
        form = registerform(request.POST)
        # check if the data in the form is valid and save the form to create new user, but don't save in database until email is verified (user is set to inactive initially)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()

            # send the verification email
            current_site = get_current_site(request) # get current site domain (needed for email template)
            mail_subject = 'Activate your account'
            message = render_to_string('verifyemail.html', {
                'user': user, # pass user object
                'domain': current_site.domain, # domain for making verification link
                'uid': urlsafe_base64_encode(force_bytes(user.pk)), # encode user's primary key
                'token': default_token_generator.make_token(user), # generate one-time use token
            })
            # get user's email address, create an email to send with a subject, message, and recipient, and then send the email
            to_email = form.cleaned_data.get('email')
            email = EmailMessage(mail_subject, message, to=[to_email])
            email.send()
            return redirect('registration_success')
    else:
        form = registerform() # otherwise, instantiate blank registration form
    return render(request, 'registerform.html', {'form': form}) # render registration form template with form instance

def activate_account(request, uidb64, token):
    try:
        uid = urlsafe_base64_decode(uidb64).decode() # decode the base64-encoded user ID
        user = User.objects.get(pk=uid) # retrieve the user from database by ID
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None # set user to none if any errors occur
    # if user exists and token is valid, activate the account and save changes to database, and redirect user to registration success page
    if user is not None and default_token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        return redirect('registration_success')
    else:
        return HttpResponse('Activation link is invalid or expired.')
def registration_success(request):
    return render(request, 'registersuccess.html')

def user_login(request):
    # retrieve username/email and password from form, and authenticate user against the database
    if request.method == 'POST':
        username_or_email = request.POST.get('username_or_email')
        password = request.POST.get('password')
        user = authenticate(request, username=username_or_email, password=password)
        # if authentication was successful (user exists and password is correct), log them in and redirect them to the preferences page
        if user is not None:
            login(request, user)
            return redirect('preferences')
        # otherwise, display error message and return to login page to try again
        else:
            messages.error(request, 'Invalid username/email or password.')
            return render(request, 'login.html', {'error_message': 'Login failed. Please try again.'})
    else:
        return render(request, 'login.html')

def login_success(request):
    return render(request, 'loginsuccess.html')

def logout_success(request):
    logout(request)
    return render(request, 'logoutsuccess.html')

class Authentication(ModelBackend): # authenticate is called with a login request, allows user to login with both email or username
    def authenticate(self, request, username=None, password=None, **kwargs):
        # try to authenticate with username
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            # if username not found, try to authenticate with email
            try:
                user = User.objects.get(email=username)
            except User.DoesNotExist: # if neither username nor email is found return none
                return None
        if user.check_password(password): # check if password is correct and return user object
            return user
        else:
            return None

def preferences(request):
    if request.method == 'POST':
        # create form instances and populate them with data from request
        preference_form = PreferenceForm(request.POST)
        career_form = CareerForm(request.POST)
        if preference_form.is_valid() and career_form.is_valid():
            # convert list of preferences and career aspirations into strings and save it in user's profile
            request.user.preferences = ', '.join(preference_form.cleaned_data['preferences'])
            request.user.career_aspirations = ', '.join(career_form.cleaned_data['career_aspirations'])
            request.user.save()  # save user's changes to database
            # after saving, reload the page with the forms filled out and a message indicating success
            return render(request, 'preferences.html', {
                'preference_form': preference_form,
                'career_form': career_form,
                'saved': True
            })
    else:
        # pre-fill the forms if the user has existing preferences/career aspirations (so if they aren't visiting this page for the first time), so they can easily edit
        preference_form = PreferenceForm(initial={
            'preferences': request.user.preferences.split(', ') if request.user.preferences else [] # if no existing preferences, preference form gets filled with nothing
        })
        career_form = CareerForm(initial={
            'career_aspirations': request.user.career_aspirations.split(', ') if request.user.career_aspirations else [] # likewise for career aspirations
        })
    return render(request, 'preferences.html', {
        'preference_form': preference_form,
        'career_form': career_form
    })


def calculate_score(request):
    user = request.user
    scores = {
        'Algorithms and Complexity': calculate_algo_comp_score(user),
        'Computer-Aided Verification': calculate_cav_score(user),
        'Computer Vision and Imaging': calculate_cvi_score(user),
        'Distributed and Dependable Systems': calculate_dds_score(user),
        'Human-Computer Interaction': calculate_hci_score(user),
        'Advanced Functional Programming': calculate_advanced_fp_score(user),
        'Advanced Networking': calculate_advanced_networking_score(user),
        'Evolutionary Computation': calculate_evocomp_score(user),
        'Intelligent Data Analysis': calculate_ida_score(user),
        'Intelligent Interactive Systems': calculate_iis_score(user),
        'Machine Learning': calculate_ML_score(user),
        'Mobile and Ubiquitous Computing': calculate_mobile_score(user),
        'Neural Computation': calculate_neural_score(user),
        'Natural Language Processing': calculate_NLP_score(user),
        'Programming Language Principles, Design, and Implementation': calculate_PLPDI_score(user),
        'Intelligent Robotics': calculate_robotics_score(user),
        'Security of Real-World Systems': calculate_security_score(user),
        'Secure Software and Hardware Systems': calculate_securesoftware_score(user),
        'Teaching Computer Science in Schools': calculate_teachingcs_score(user)
    }


    module_descriptions = {
        'Algorithms and Complexity': "Algorithms are at the heart of computer science. In this module we will develop a range of core algorithmic ideas such as dynamic programming, greedy methods, divide-and-conquer techniques, and network flows. We will then learn how to use these to design efficient algorithms for a range of problems, motivated by a range of applications. We will then consider core concepts from computational complexity theory such as NP-completeness, and their implications for algorithm design. Finally, we will consider some advanced modern topics such as approximate and randomized algorithms, parameterized algorithms and complexity, and algorithms for streams of data.",
        'Computer-Aided Verification': "Showing that computer systems, hardware or software, are free of bugs is an important and challenging area of computer science but is essential in contexts such as safety-critical applications or computer security, where the consequences can be severe. This module introduces the field of formal verification, which rigorously checks the correctness of computer systems. Students will be introduced to the concept of mathematical modelling of systems, both sequential and parallel, learn how to formalise correctness properties using logics, notably temporal logic, and how to verify them automatically using techniques such as model checking. The module will cover both the theory and algorithms underlying these verification techniques and give a practical introduction to using verification tools.",
        'Computer Vision and Imaging': "Vision is one of the major senses that enables humans to act (and interact) in (ever)changing environments, and imaging is the means by which we record visual information in a form suitable for computational processing. Together, imaging and computer vision play an important role in a wide range of intelligent systems, from advanced microscopy techniques to autonomous vehicles. This module will focus on the fundamental computational principles that enable an array of picture elements, acquired by one of a multitude of imaging technologies, to be converted into structural and semantic entities necessary to understand the content of images and to accomplish various perceptual tasks. We will study the problems of image formation, low level image processing, object recognition, categorisation, segmentation, registration, stereo vision, motion analysis, tracking and active vision. The lectures will be accompanied by a series of exercises in which these computational models will be designed, implemented and tested in real-world scenarios.",
        'Distributed and Dependable Systems': "Distributed systems have become commonplace, with such systems providing the vast majority of services that we have come to depend on every day. This module studies a range of topics in distributed systems from a practical and theoretical perspective. Students will learn how to analyse, design and implement efficient, fault tolerant solutions to modern problems throughout a rigorous understanding of classical approaches and results.",
        'Human-Computer Interaction': "Human-computer interaction (HCI) explores the technical and psychological issues arising from the interface between people and machines. Understanding HCI is essential for designing effective hardware and software user interfaces. This module teaches the theory and practice of HCI methodologies for both design and evaluation.",
        'Advanced Functional Programming': "This module exposes students to state of the art functional programming languages and their mathematical foundations in the lambda calculus and type theory. Students can expect to develop advanced functional programming skills and awareness of experimental programming languages.",
        'Evolutionary Computation': "Evolutionary algorithms (EAs) are a class of optimisation techniques drawing inspiration from principles of biological evolution. They typically involve a population of candidate solutions from which the better solutions are selected, recombined, and mutated to form a new population of candidate solutions. This continues until an acceptable solution is found. Evolutionary algorithms are popular in applications where no problem-specific method is available, or when gradient-based methods fail. They are suitable for a wide range of challenging problem domains, including dynamic and noisy optimisation problems, constrained optimisation problems, and multi-objective optimisation problems. EAs are used in a wide range of disciplines, including optimisation, engineering design, machine learning, financial technology (“fintech”), and artificial life. In this module, we will study the fundamental principles of evolutionary computation, a range of different EAs and their applications, and a selection of advanced topics which may include time-complexity analysis, neuro-evolution, co-evolution, model-based EAs, and modern multi-objective EAs.",
        'Intelligent Data Analysis': "The ‘information revolution’ has generated large amounts of data, but valuable information is often hidden and hence unusable. In addition, the data may come in many different forms, e.g. high-dimensional data collections, stream and time-series data, textual documents, images, large-scale graphs representing communication in social networks or protein-to-protein interactions etc. This module will introduce a range of techniques in the fields of pattern analysis, data analytics and data mining that aim to extract hidden patterns and useful information in order to understand such challenging data.",
        'Intelligent Interactive Systems': "Computer systems are increasingly designed to cooperate with people. For example, semi-autonomous vehicles automate part of the problem of driving but leave overall control with a human driver. Similarly, medical systems make use of vast amounts of data and the latest machine learning but must work with doctors to determine a diagnosis. Similarly, decision supports systems, tutoring systems, dialogue systems, and recommender systems. These Intelligent Interactive Systems, and the theory of human psychology that underpins them are the subject of this module. This is an area of computer science that is making rapid progress with new theory and methods influencing real-world systems. It is an area that is driven, in part, by the vast amount of data concerning human behaviour that is now routinely collected and the desire to not simply classify it but to understand it. The module will introduce students to this new and exciting topic and to the methods needed to build Intelligent Interactive Systems. There will be a strong focus on algorithms for modelling and understanding people using computers.",
        'Machine Learning': "Machine learning studies how computers can autonomously learn from available data, without being explicitly programmed. The module will provide a solid foundation to machine learning by giving an overview of the core concepts, theories, methods, and algorithms for learning from data. The emphasis will be on the underlying theoretical foundations, illustrated through a set of methods used in practice. This will provide the student with a good understanding of how, why and when various machine learning methods work.",
        'Mobile and Ubiquitous Computing': "This module is concerned with the issues surrounding mobile and ubiquitous computing systems. It examines the particular issues that arise in these systems both from a technical perspective and in terms of usability and interaction. The underlying theoretical and technological frameworks are discussed. Students are introduced to development tools and techniques for building mobile systems and their understanding is reinforced through practical work. The present and potential future applications are reviewed.",
        'Neural Computation': "This course focuses on artificial neural networks and their use in machine learning. It covers the fundamental underlying theory, as well as methodologies for constructing modern deep neural networks, which nowadays have practical applications in a variety of industrial and research domains. The course also provides practical experience of designing and implementing a neural network for a real-world application.",
        'Natural Language Processing': "Natural Language Processing enables computers to understand and reason about human languages such as English and has resulted in many exciting technologies such as conversational assistants, machine translation and (intelligent) internet search. This module would provide the theoretical foundations of NLP as well as applied techniques for extracting and reasoning about information from text. The module explores three major themes: Computational Models of human cognition such as memory, attention and psycholinguistics Symbolic AI methods for processing language such as automated reasoning, planning, parsing of grammar, and conversational systems. Statistical Models of Language including the use of machine learning to infer structure and meaning.",
        'Programming Language Principles, Design, and Implementation': "By now students will have seen and used a variety of programming languages. In this module they will also understand the principles behind their design along with techniques for transforming human-friendly programs written in high-level programming languages such as C, Java, or Haskell, into machine-friendly sets of instructions, written for example in assembly. This module introduces some of the central concepts and techniques used to design and study programming languages, from syntactic and semantic specification to compilation. Students will see how to design a language by first defining its syntax (e.g., grammar rules) and semantics (e.g., operational semantics). They will further learn how types can be used to guarantee that programs are “safe”, thereby preventing certain catastrophic errors. We will see that types have other uses. They can for example be used as abstraction mechanisms that allow hiding implementation details. Finally, the module will describe the structure of a typical compiler, including front end phases that implement the syntactic and semantic specification of a language, and the code generation and optimization backend phases, as well as the key techniques used in those phases.",
        'Intelligent Robotics': "Artificial Intelligence is concerned with the design and use of computer systems to understand and mimic human-level decision making. These systems represent, reason with, and learn from, different descriptions of knowledge and uncertainty. In this module we will address these issues in the context of intelligent mobile robots. The lectures will teach theories of perception, estimation, prediction, decision- making, learning and control, all from the perspective of robotics. In the laboratory sessions students will implement some of these theories on simulated or real robots to see how theory can be applied in practice.",
        'Security of Real-World Systems': "Building on Security and Networks, this module teaches how to find, analyse, and mitigate security vulnerabilities in real-world systems. It will also teach students how to assess the threats to a system, and how to protect against them. A range of practical analysis methods and tools are covered.",
        'Secure Software and Hardware Systems': "Note: This module is for 4th year Masters students only. This module covers the principles of software and hardware security. Classic design principles for the protection of information in computer systems are introduced. Some of the most important vulnerabilities in current software and hardware systems and the corresponding attacks will be reviewed, and tools and techniques for analysing and defending against them will be studied.",
        'Teaching Computer Science in Schools': "The module is the implementation of the national Undergraduate Ambassador Scheme (UAS) by the School of Computer Science at the University of Birmingham. Its design follows closely the recommendations and guidelines of the UAS. The module will reward students with course credit for working as a student-tutor with teachers in local schools during semester 2 with training sessions during semester 1. Students should note that while the placements take place during semester 2, the training sessions are in semester 1. It is not possible to join this module without taking part in the training sessions. Students will learn about the key issues affecting school education today. They will have the satisfaction of making a positive impact on the education of pupils of all ages and the chance to act as a role model for computer science. It is a chance to put something back into the community by sharing knowledge and helping to motivate young people and raise their aspirations towards computer science. They will develop confidence in answering questions about their own subject and in devising appropriate ways to communicate a difficult principle or concept. They will develop communication skills and gain a better understanding of their own level of expertise. For those of who are interested in teaching as a profession, this will be an opportunity to explore whether it is a path they want to pursue. They will learn to devise and develop computer science projects and teaching methods appropriate to engage the relevant age group. The module will involve: Attending at 8 two-hour training sessions giving students an introduction to the fundamentals of working with children and conduct in the school environment. Undergoing a Disclosure and Barring Service (DBS) check prior to entering the classroom. Working in pairs with a specific teacher at a local school. Spending one full day a week in school for 9 weeks. Designing and implementing an extra-curricular project (agreed with the teacher)",
        'Advanced Networking': "One of the defining characteristics of today’s computer systems is their ability to exchange information. Whether we are talking about the smallest home network or the Internet as a whole, computer networks play a key role in many computer applications. An enormous number of applications, from general services such as the World Wide Web to specialised messaging or video streaming apps rely on networks and the common standards and protocols which make them work. This module introduces the basic concepts, technologies, architectures and standards involved in computer networks, together with methods for their design and implementation. This will include discussion of data transmission protocols, TCP/IP, LANs and WANs, communication mechanisms and synchronization issues. The module will be based on the discussion of real-world case studies, research papers and standardisation documents.",
    }

    sorted_scores = {}  # dictionary to store the scores sorted from highest to lowest

    # sort the scores dictionary by values in descending order
    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    for key in sorted_keys:
        sorted_scores[key] = scores[key]  # populate the scores dictionary


    return render(request, 'calculate_score.html', {'sorted_scores': sorted_scores, 'module_descriptions': module_descriptions})


def submit_feedback(request): # feedback form to submit feedback for a specific module
    if request.method == 'POST':
        form = FeedbackForm(request.POST)
        if form.is_valid():
            module = form.cleaned_data['module']
            feedback = form.cleaned_data['feedback']
            if module == 'Algorithms and Complexity':
                algocomp_comments.append(feedback) # add feedback to the list of comments for specified module
            return redirect('feedback_success')
    else:
        form = FeedbackForm()
    return render(request, 'feedback_form.html', {'form': form})

def feedback_success(request):
    return render(request, 'feedback_success.html')



