from django import forms
from django.contrib.auth.forms import UserCreationForm
from OptiMods.models import User


class registerform(UserCreationForm):
    email = forms.EmailField(max_length=100, help_text='Required. Enter a valid email address.'),
    first_name = forms.CharField(max_length=30, required=True)
    last_name = forms.CharField(max_length=30, required=True)
    username = forms.CharField(max_length=150, required=True)

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']

class PreferenceForm(forms.Form):
    preferences = forms.MultipleChoiceField(choices=(
        ('Maths', 'I enjoy maths and would like to do a module with maths in it'),
        ('Security', 'I like the idea of modules that involve security'),
        ('Complexity', 'I like the idea of modules that involve complexity theory'),
        ('Algorithms', 'I like the idea of modules that involve algorithms'),
        ('Physics', 'I like the idea of modules that involve physics'),
        ('Logic', 'I like the idea of modules that involve logic (e.g first year logic module)'),
        ('Functional Programming', 'I enjoyed Functional Programming in 2nd year'),
        ('Networking', 'I like the idea of modules that involve networking (computer networks, data transmission protocols, communication mechanisms, cloud systems etc)'),
        ('Formal Verification', 'I like the idea of modules that involve formal verification (checking correctness of computer systems, expressing formal correctness properties of systems using logic)'),
        ('MATLAB', 'I like the idea of modules that involve the use of MATLAB'),
        ('Machine Learning', 'I like the idea of modules that involve learning about Machine Learning'),
        ('Computer Vision', 'I like the idea of modules that teach the concept of how computers process images, how computers perceive things, e.g object recognition'),
        ('Distributed Systems', 'I like the idea of modules that teach the concept of distributed systems (and the idea of fault tolerance'),
        ('Evolutionary Algorithms', 'I like the idea of modules that teach the concept of evolutionary algorithms (a class of optimisation techniques drawing inspiration from principles of biological evolution)'),
        ('Psychology', 'I like the idea of modules that involve psychology (more specifically, human cognition and behaviour in the context of technology interaction.)'),
        ('Design Problems', 'I like the idea of modules that involve design problems'),
        ('Python', 'I like the idea of modules that involve the use of Python'),
        ('Robotics', 'I like the idea of modules that involve Robotics'),
        ('Data Analysis', 'I like the idea of modules that teach about things like pattern analysis, data analytics and data mining'),
        ('Mobile', 'I like the idea of modules that teach about building/developing mobile systems'),
        ('Artificial Intelligence', 'I like the idea of modules that involve artificial intelligence'),
        ('NLP', 'I like the idea of modules that teach about NLP (Natural Language Processing), how computers understand languages'),
        ('Artificial Neural Networks', 'I like the idea of modules that teach about artificial neural networks and their use in machine learning'),
        ('Programming', 'I like the idea of modules that teach about the principles of a programming language and how programming languages actually work'),
        ('Teaching', 'I like the idea of a module that focuses on training you to teach Computer Science at schools'),
    ),  widget = forms.CheckboxSelectMultiple, required = False)

class CareerForm(forms.Form):
    career_aspirations = forms.MultipleChoiceField(choices=(
        ('Software Engineer', 'Software Engineer'),
        ('Data Scientist', 'Data Scientist'),
        ('Machine Learning', 'Machine Learning (e.g Machine Learning Engineer'),
        ('Game Developer', 'Game Developer'),
        ('Cybersecurity', 'Cybersecurity'),
        ('Quantum Related', 'Quantum Related'),
        ('Quantitative Analyst', 'Quantitative Analyst'),
        ('Networking', 'Networking (e.g Network Engineer, Network Security Analyst, Cloud Computing etc)'),
        ('Verification Engineer', 'Verification Engineer'),
        ('Computer Vision Engineer', 'Computer Vision Engineer'),
        ('Software Developer', 'Software Developer'),
        ('UX Designer', 'UX (User Experience) Designer'),
        ('AI/ML Engineer', 'AI/ML Engineer'),
        ('Robotics Engineer', 'Robotics Engineer'),
        ('Data Analyst', 'Data Analyst'),
        ('Mobile App Developer', 'Mobile App Developer'),
        ('Compiler Engineer', 'Compiler Engineer'),
        ('Security Engineer', 'Security Engineer'),
        ('CS Teacher', 'CS Teacher'),
    ),  widget = forms.CheckboxSelectMultiple, required = False)

class FeedbackForm(forms.Form):
    module_choices = [
        ('Algorithms and Complexity', 'Algorithms and Complexity'),
        ('Computer-Aided Verification', 'Computer-Aided Verification'),
        ]

    module = forms.ChoiceField(choices=module_choices, label='Module')
    feedback = forms.CharField(widget=forms.Textarea, label='Feedback')