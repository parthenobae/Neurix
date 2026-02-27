from flask import render_template, request, Blueprint
from neurix.models import Post

main = Blueprint('main', __name__)


@main.route("/")
@main.route("/home")
def home():
    return render_template('home.html')

@main.route('/resources')
def resources():
    return render_template('resources.html', title='Resources')

@main.route("/blogs")
def blogs():
    page = request.args.get('page', 1, type=int)
    posts = Post.query.order_by(Post.date_posted.desc()).paginate(page=page, per_page=5)
    return render_template('blogs.html', title='Blogs', posts=posts)
