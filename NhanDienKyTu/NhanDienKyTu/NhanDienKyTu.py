
from turtle import Turtle, Screen
import turtle
import svgwrite
from PIL import Image

MOVING, DRAGGING = range(2)  # states
from Predict import RandomForestPredict
yertle = Turtle()

def move_handler(x, y):
    if state != MOVING:  # ignore stray events
        return

    onmove(yertle.screen, None)  # avoid overlapping events
    yertle.penup()
    yertle.setheading(yertle.towards(x, y))
    yertle.goto(x, y)
    onmove(yertle.screen, move_handler)

def click_handler(x, y):
    global state

    yertle.onclick(None)  # disable until release
    onmove(yertle.screen, None)  # disable competing handler

    yertle.onrelease(release_handler)  # watch for release event
    yertle.ondrag(drag_handler)  # motion is now dragging until release

    state = DRAGGING

def release_handler(x, y):
    global state

    yertle.onrelease(None)  # disable until click
    yertle.ondrag(None)  # disable competing handler

    yertle.onclick(click_handler)  # watch for click event
    onmove(yertle.screen, move_handler)  # dragging is now motion until click

    state = MOVING

def drag_handler(x, y):
    if state != DRAGGING:  # ignore stray events
        return

    yertle.ondrag(None)  # disable event inside event handler
    yertle.pendown()
    yertle.setheading(yertle.towards(x, y))
    yertle.goto(x, y)
    yertle.ondrag(drag_handler)  # reenable event on event handler exit

def ScreenShot():
    yertle.hideturtle()
    yertle.screen.getcanvas().postscript(file="char.eps", colormode='color')
    image_eps = 'char.eps'
    im = Image.open(image_eps)

    fig = im.convert('RGBA')
    image_png= 'char.png'
    fig.save(image_png, lossless = True)

    resize_image("char.png")
    #jpg = Image.open("char.png")
    #rgb_im = jpg.convert('RGB')
    #rgb_im.save("char.jpg")
    yertle.showturtle()
    yertle.clear()
    print('Ok')
    predict('char.png')

def predict(image):
    ef_predict = RandomForestPredict()
    ef_predict.predict(image)


def onmove(self, fun, add=None):

    if fun is None:
        self.cv.unbind('<Motion>')
    else:
        def eventfun(event):
            fun(self.cv.canvasx(event.x) / self.xscale, -self.cv.canvasy(event.y) / self.yscale)
        self.cv.bind('<Motion>', eventfun, add)

def resize_image(image_name):
    image = Image.open(image_name)
    image.thumbnail((28, 28))
    image.save('char.png')
    #image.show()

yertle.screen.setup(500, 500)
yertle.screen.screensize(500, 500)

yertle.pensize(30)
yertle.turtlesize(0.2)
yertle.speed('fastest')
state = MOVING 

# Initially we track the turtle's motion and left button clicks
onmove(yertle.screen, move_handler)  # a la screen.onmove(move_handler)
yertle.onclick(click_handler)  # a click will turn motion into drag

yertle.screen.listen()
yertle.screen.onkeypress(ScreenShot, 'Return')

yertle.screen.mainloop()

