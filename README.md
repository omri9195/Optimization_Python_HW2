Hey :)

Tested on python version 3.9.13. To avoid compatibility issues please use this python version. With high probability this will work well with other python versions as well.
To check python version run 'python --version'

In order to run the project please follow these steps:
1. Open terminal, clone the project as follows:
    'git clone https://github.com/omri9195/Optimization_Python_HW2.git'
2. Go to appropriate directory:
    'cd Optimization_Python_HW2'
3. Set up virtual environment as follows:
   <br> Mac (Tested on Mac): <br>
         'python3 -m venv venv'<br>
         'source venv/bin/activate'
   <br> 
  Windows (Not tested on windows):<br>
         'python -m venv venv'<br>
         'venv\Scripts\activate'
5. Install dependencies:
     'pip install -r requirements.txt'
6. Run the project with either one of:<br>
     'python -m tests.test_unconstrained_min' <br>
     'python -m tests.test_constrained_min'


You may observe iteration details in console (terminal), the plots will be created in a plots folder by constrained or unconstrained, and then by function name as visible in this directory.

Thanks :)
