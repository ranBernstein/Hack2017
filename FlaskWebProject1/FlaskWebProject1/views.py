"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from FlaskWebProject1 import app

@app.route('/')
@app.route('/home')
def home():
	import subprocess
	p1 = subprocess.Popen(["Release\ConsoleApplication1"],stdout=subprocess.PIPE)
	token = p1.communicate()[0].strip()
	return render_template(
		'index.html',
		title='Home Page',
		year=datetime.now().year,
		token=token
	)

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Contact',
        year=datetime.now().year,
        message='Your contact page.'
    )

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='About',
        year=datetime.now().year,
        message='Your application description page.'
    )
