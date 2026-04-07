import random
import ctypes
import arcade
from arcade import gui
from EndGame import GameOverView
from EndGame import scoreSaving

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# Constants
SCREEN_WIDTH = screensize[0]
SCREEN_HEIGHT = screensize[1]
TUBE_HEIGHT = 300
TUBE_WIDTH = 50
TUBE_CENTER_X1 = SCREEN_WIDTH / 4
TUBE_CENTER_X2 = 3 * SCREEN_WIDTH / 4
TUBE_CENTER_X3 = TUBE_CENTER_X1 + 75
TUBE_CENTER_X4 = TUBE_CENTER_X2 - 75
TUBE_CENTER_Y = SCREEN_HEIGHT / 2
POINT_LEFT_X = SCREEN_WIDTH / 6
POINT_RIGHT_X = 5 * SCREEN_WIDTH / 6
POINT_UP_Y = 5 * SCREEN_HEIGHT / 6
POINT_DOWN_Y = SCREEN_HEIGHT / 6
HEIGHT_BAR1 = TUBE_CENTER_Y - TUBE_HEIGHT / 2 + TUBE_HEIGHT / 6
HEIGHT_BAR2 = TUBE_CENTER_Y + TUBE_HEIGHT / 2 - TUBE_HEIGHT / 6
HEIGHT_BAR1_THRESHOLD = TUBE_HEIGHT / 6
HEIGHT_BAR2_THRESHOLD = 5 * TUBE_HEIGHT / 6

COLORS = ["red", "yellow", "green", "brown"]
arcadeColors = {
    "red": arcade.color.RED,
    "yellow": arcade.color.YELLOW,
    "brown": arcade.color.BROWN,
    "green": arcade.color.GREEN,
}

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
STOP = 4


def maybe_stop(probab):
    """Function to triggers the events of the game

    :param probab: probability of an event triggering
    :return:
    """
    proba = random.uniform(0.1, 1000.0)
    if proba <= probab:
        return True
    else:
        return False


"""
Main game window and management
"""


class GameView(arcade.View):
    def __init__(self, level_number, first_name, last_name):
        super().__init__()

        """
        Stocks the infos from previous views
        """
        self.level = level_number
        self.player_first_name = first_name
        self.player_last_name = last_name

        """
        Initialisation of the score for different events
        """
        self.score = {
            "tubes": 0,
            "circle": 0,
            "questions": 0,
            "lights": 0,
        }

        """
        Used for the math question event
        """
        self.sum = None

        """
        Level management
        """
        if level_number == 1:
            self.feature_circle = False
            self.feature_labels = False
            self.feature_lights = False
            self.feature_more_tubes = False
        if level_number == 2:
            self.feature_circle = True
            self.feature_labels = True
            self.feature_lights = True
            self.feature_more_tubes = False
        if level_number == 3:
            self.feature_circle = True
            self.feature_labels = True
            self.feature_lights = True
            self.feature_more_tubes = True
        """
        Timer management
        """
        self.total_time = 0.0

        self.timer_text = arcade.Text(
            text="00:00:00",
            start_x=9 * SCREEN_WIDTH / 10,
            start_y=9 * SCREEN_HEIGHT / 10,
            color=arcade.color.WHITE,
            font_size=20,
        )

        """
        Score management
        """
        self.question_math_on = False
        self.question_color_on = False
        self.lights_on = False

        """
        Tubes controls
        """
        self.tube1_adv = True
        self.tube2_adv = True
        self.counter1 = TUBE_HEIGHT / 2
        self.counter2 = TUBE_HEIGHT / 2

        if self.feature_more_tubes:
            self.tube3_adv = True
            self.tube4_adv = True
            self.counter3 = TUBE_HEIGHT / 2
            self.counter4 = TUBE_HEIGHT / 2

        """
        Circle mouvement controls
        """
        self.circle_mvt_state = RIGHT
        self.remember_previous_state = RIGHT

        """
        Ui Manager creation : we can add widgets into it to use them in the view
        """
        self.ui_manager = arcade.gui.UIManager()
        self.ui_manager.enable()

        """
        End game button
        """
        self.end_button = arcade.gui.UIFlatButton(10, SCREEN_HEIGHT - 100, text="End Game")
        self.end_button.on_click = self.end_game
        self.ui_manager.add(self.end_button)

        """
        Creation of the buttons dealing with the tubes
        """
        tube_button1 = arcade.gui.UIFlatButton(TUBE_CENTER_X1 - TUBE_WIDTH / 2, TUBE_CENTER_Y - TUBE_HEIGHT / 2 - 60,
                                               50, 50)
        tube_button1.on_click = self.on_button1_click
        self.ui_manager.add(  # function to add to the ui manager
            tube_button1
        )
        tube_button2 = arcade.gui.UIFlatButton(TUBE_CENTER_X2 - TUBE_WIDTH / 2, TUBE_CENTER_Y - TUBE_HEIGHT / 2 - 60,
                                               50, 50)
        tube_button2.on_click = self.on_button2_click
        self.ui_manager.add(
            tube_button2
        )

        if self.feature_more_tubes:  # Level 3 only
            tube_button3 = arcade.gui.UIFlatButton(TUBE_CENTER_X3 - TUBE_WIDTH / 2,
                                                   TUBE_CENTER_Y - TUBE_HEIGHT / 2 - 60,
                                                   50, 50)
            tube_button3.on_click = self.on_button3_click
            self.ui_manager.add(  # function to add to the ui manager
                tube_button3
            )
            tube_button4 = arcade.gui.UIFlatButton(TUBE_CENTER_X4 - TUBE_WIDTH / 2,
                                                   TUBE_CENTER_Y - TUBE_HEIGHT / 2 - 60,
                                                   50, 50)
            tube_button4.on_click = self.on_button4_click
            self.ui_manager.add(
                tube_button4
            )

        """
        Creation of a sprite list, to append future sprites
        """
        self.sprite_list = arcade.SpriteList()

        """
        Circle creation
        """
        self.circle_sprite = arcade.SpriteCircle(50, arcade.color.RED)
        self.circle_sprite.center_x = POINT_LEFT_X
        self.circle_sprite.center_y = POINT_UP_Y
        self.sprite_list.append(self.circle_sprite)  # add sprite to list

        """
        Creation of the 3 centers widgets
        """
        self.label1 = arcade.Text(  # First text label for colors
            text="Color Here",
            start_x=SCREEN_WIDTH / 2 - 50,
            start_y=SCREEN_HEIGHT / 2 + 50,
            color=arcade.color.YELLOW,
            width=350,
            font_size=20,
        )
        self.input_field = arcade.gui.UIInputText(  # Input for the user
            color=arcade.color.BROWN,
            font_size=24,
            text='HERE',
            text_color=arcade.color.GOLD
        )
        self.label2 = arcade.Text(  # Second text label for math questions
            text="Math here",
            start_x=SCREEN_WIDTH / 2 - 50,
            start_y=SCREEN_HEIGHT / 2 - 50,
            color=arcade.color.YELLOW,
            font_size=24
        )

        """
        Creation of the two upper lights sprites
        """

        self.h_box_up = arcade.gui.UIBoxLayout(vertical=False, space_between=64)
        self.up_light1_label = arcade.gui.UILabel(width=50, height=50, text="HERE")
        self.up_light2_label = arcade.gui.UILabel(width=50, height=50, text="HERE")
        self.h_box_up.add(child=self.up_light1_label)
        self.h_box_up.add(child=self.up_light2_label)

        self.up_light1 = arcade.SpriteSolidColor(60, 60, arcade.color.BLACK)
        self.up_light2 = arcade.SpriteSolidColor(60, 60, arcade.color.BLACK)

        """
        Creation of a vertical box the horizontal boxes and widgets
        """
        self.v_box = arcade.gui.UIBoxLayout(space_between=75)
        self.v_box.add(self.h_box_up)
        # self.v_box.add(self.label1)
        self.v_box.add(self.input_field)
        # self.v_box.add(self.label2)

        """
        Creation of the buttons to interact with the lights
        """
        self.button_down1 = arcade.gui.UIFlatButton(width=50, height=50)
        self.button_down1.on_click = self.on_click_button_left
        self.button_down2 = arcade.gui.UIFlatButton(width=50, height=50)
        self.button_down2.on_click = self.on_click_button_right

        """
        Creation of a horizontal box to store and aligns the two buttons
        """
        self.h_box_down = arcade.gui.UIBoxLayout(vertical=False, space_between=64)

        self.h_box_down.add(
            child=self.button_down1,
        )
        self.h_box_down.add(
            child=self.button_down2,
        )

        """
        We can add the horizontal box to the vertical one
        """
        self.v_box.add(
            child=self.h_box_down,
        )

        self.ui_manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x",
                anchor_y="center_y",
                child=self.v_box
            )
        )

        # rect

        """
        Sprites that make the colors for the two down buttons
        """
        self.rectangle1 = arcade.SpriteSolidColor(60, 60, arcade.color.GREEN)
        self.rectangle2 = arcade.SpriteSolidColor(60, 60, arcade.color.RED)

        """
        Center the rectangle sprites on the buttons and lights
        """

        self.sprite_list.append(self.rectangle1)
        self.sprite_list.append(self.rectangle2)
        self.sprite_list.append(self.up_light1)
        self.sprite_list.append(self.up_light2)

    """
    Function to draw every objects we need
    """

    def on_draw(self):

        self.clear()  # Clear the menu

        arcade.set_background_color(arcade.color.BLACK)  # sets the background color

        """
        Draw the outline of the tubes
        """
        arcade.draw_rectangle_outline(TUBE_CENTER_X1, TUBE_CENTER_Y, TUBE_WIDTH, TUBE_HEIGHT, arcade.color.RED, 4)
        arcade.draw_rectangle_outline(TUBE_CENTER_X2, TUBE_CENTER_Y, TUBE_WIDTH, TUBE_HEIGHT, arcade.color.BLUE, 4)
        if self.feature_more_tubes:
            arcade.draw_rectangle_outline(TUBE_CENTER_X3, TUBE_CENTER_Y, TUBE_WIDTH, TUBE_HEIGHT, arcade.color.GREEN, 4)
            arcade.draw_rectangle_outline(TUBE_CENTER_X4, TUBE_CENTER_Y, TUBE_WIDTH, TUBE_HEIGHT, arcade.color.VIOLET,
                                          4)

        """
        Positions of the colored rectangles, for the two low buttons and the 2 upper lights
        """
        self.rectangle1.set_position(self.button_down1.center_x, self.button_down1.center_y)
        self.rectangle2.set_position(self.button_down2.center_x, self.button_down2.center_y)

        self.up_light1.set_position(self.up_light1_label.center_x, self.up_light1_label.center_y)
        self.up_light2.set_position(self.up_light2_label.center_x, self.up_light2_label.center_y)

        """
        Draw every widgets inside the uimanager
        """
        self.ui_manager.draw()
        self.sprite_list.draw()
        self.timer_text.draw()
        self.label1.draw()
        self.label2.draw()

        """
        Draw lines to fill the tubes, using the counter increment
        """
        arcade.draw_rectangle_filled(TUBE_CENTER_X1, TUBE_CENTER_Y - TUBE_HEIGHT / 2 + self.counter1 / 2, TUBE_WIDTH,
                                     self.counter1, arcade.color.RED)
        arcade.draw_rectangle_filled(TUBE_CENTER_X2, TUBE_CENTER_Y - TUBE_HEIGHT / 2 + self.counter2 / 2, TUBE_WIDTH,
                                     self.counter2, arcade.color.BLUE)
        if self.feature_more_tubes:  # only on level 3
            arcade.draw_rectangle_filled(TUBE_CENTER_X3, TUBE_CENTER_Y - TUBE_HEIGHT / 2 + self.counter3 / 2,
                                         TUBE_WIDTH,
                                         self.counter3, arcade.color.GREEN)
            arcade.draw_rectangle_filled(TUBE_CENTER_X4, TUBE_CENTER_Y - TUBE_HEIGHT / 2 + self.counter4 / 2,
                                         TUBE_WIDTH,
                                         self.counter4, arcade.color.VIOLET)

        """
        Drawn lines for the path of the circle
        """
        arcade.draw_line(POINT_LEFT_X, POINT_UP_Y, POINT_RIGHT_X, POINT_UP_Y, arcade.color.WHITE)
        arcade.draw_line(POINT_RIGHT_X, POINT_UP_Y, POINT_RIGHT_X, POINT_DOWN_Y, arcade.color.WHITE)
        arcade.draw_line(POINT_RIGHT_X, POINT_DOWN_Y, POINT_LEFT_X, POINT_DOWN_Y, arcade.color.WHITE)
        arcade.draw_line(POINT_LEFT_X, POINT_DOWN_Y, POINT_LEFT_X, POINT_UP_Y, arcade.color.WHITE)

        """
        Draw the 2 two lines of warning for the 2 tubes
        """

        arcade.draw_line(TUBE_CENTER_X1 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR1,
                         TUBE_CENTER_X1 + TUBE_WIDTH / 2 + 1,
                         HEIGHT_BAR1, arcade.color.GOLD, line_width=3)

        arcade.draw_line(TUBE_CENTER_X1 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR2,
                         TUBE_CENTER_X1 + TUBE_WIDTH / 2 + 1,
                         HEIGHT_BAR2, arcade.color.GOLD, line_width=3)

        arcade.draw_line(TUBE_CENTER_X2 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR1,
                         TUBE_CENTER_X2 + TUBE_WIDTH / 2 + 1,
                         HEIGHT_BAR1, arcade.color.GOLD, line_width=3)

        arcade.draw_line(TUBE_CENTER_X2 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR2,
                         TUBE_CENTER_X2 + TUBE_WIDTH / 2 + 1,
                         HEIGHT_BAR2, arcade.color.GOLD, line_width=3)

        if self.feature_more_tubes:
            arcade.draw_line(TUBE_CENTER_X3 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR1,
                             TUBE_CENTER_X3 + TUBE_WIDTH / 2 + 1,
                             HEIGHT_BAR1, arcade.color.GOLD, line_width=3)

            arcade.draw_line(TUBE_CENTER_X3 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR2,
                             TUBE_CENTER_X3 + TUBE_WIDTH / 2 + 1,
                             HEIGHT_BAR2, arcade.color.GOLD, line_width=3)

            arcade.draw_line(TUBE_CENTER_X4 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR1,
                             TUBE_CENTER_X4 + TUBE_WIDTH / 2 + 1,
                             HEIGHT_BAR1, arcade.color.GOLD, line_width=3)

            arcade.draw_line(TUBE_CENTER_X4 - TUBE_WIDTH / 2 - 1, HEIGHT_BAR2,
                             TUBE_CENTER_X4 + TUBE_WIDTH / 2 + 1,
                             HEIGHT_BAR2, arcade.color.GOLD, line_width=3)

    def score_controller(self, delta_time):
        """Increments the corresponding score depending on the events

        :param delta_time: time measure, used to increment the score
        :return:
        """
        if self.check_tubes_position():
            self.score["tubes"] += delta_time
        if self.circle_mvt_state == STOP:
            self.score["circle"] += delta_time
        if self.question_color_on or self.question_math_on:
            self.score["questions"] += delta_time
        if self.lights_on:  # not working
            pass

    def on_update(self, delta_time: float):
        if self.feature_circle:
            self.control_circle()  # Makes the circle move
        self.adv_click()  # Makes the tube fill up or not
        if self.feature_labels and self.feature_lights:
            if maybe_stop(1.0):
                self.update_text1()  # Make the first label change text
            if maybe_stop(1.0):
                self.update_text2()  # Make the second label change text
            if maybe_stop(1.0):
                self.update_button_left()
            if maybe_stop(1.0):  # Make the left light change color
                self.update_button_right()  # Make the right light change color
        self.score_controller(delta_time)
        """
        All the logic to move, and the game logic goes here. Found on internet
        """
        # Accumulate the total time
        self.total_time += delta_time

        # Calculate minutes
        minutes = int(self.total_time) // 60

        # Calculate seconds by using a modulus (remainder)
        seconds = int(self.total_time) % 60

        # Use string formatting to create a new text string for our timer
        self.timer_text.text = f"{minutes:02d}:{seconds:02d}"
        if self.timer_text.text == "10:00":  # Game ends on 10 minutes marker
            self.end_game()

    def adv_click(self):
        """Controls the counters increment for the tubes

        :return:
        """
        if TUBE_HEIGHT > self.counter1 > -1:  # stops the counters if the tubes fills too much or is empty
            if self.tube1_adv:
                self.counter1 += 0.5
            else:
                self.counter1 -= 0.5
        if TUBE_HEIGHT > self.counter2 > -1:
            if self.tube2_adv:
                self.counter2 += 0.3
            else:
                self.counter2 -= 0.3
        """
        if self.counter1 < TUBE_HEIGHT/6 or self.counter1 > 5*TUBE_HEIGHT/6 or self.counter2 < TUBE_HEIGHT/6 or self.counter2 > 5*TUBE_HEIGHT/6:
            self.tube_defeat = True
        else:
            self.tube_defeat = False
        """

        if self.feature_more_tubes:  # Only on level 3
            if TUBE_HEIGHT > self.counter3 > -1:
                if self.tube3_adv:
                    self.counter3 += 0.5
                else:
                    self.counter3 -= 0.5
            if TUBE_HEIGHT > self.counter4 > -1:
                if self.tube4_adv:
                    self.counter4 += 0.3
                else:
                    self.counter4 -= 0.3
            """
            if self.counter3 < TUBE_HEIGHT / 6 or self.counter3 > 5 * TUBE_HEIGHT / 6 or self.counter4 < TUBE_HEIGHT / 6 or self.counter4 > 5 * TUBE_HEIGHT / 6:
                self.tube_defeat = True
            else:
                self.tube_defeat = False
            """

    def check_tubes_position(self):
        """Checks the state of the tubes filling up and tells the score increment for the tubes to increase of case of
        filling up too much or too little

        :return:
        """
        if HEIGHT_BAR2_THRESHOLD > self.counter1 > HEIGHT_BAR1_THRESHOLD and HEIGHT_BAR2_THRESHOLD > self.counter2 > HEIGHT_BAR1_THRESHOLD:
            if self.level == 3:
                if HEIGHT_BAR2_THRESHOLD > self.counter3 > HEIGHT_BAR1_THRESHOLD and HEIGHT_BAR2_THRESHOLD > self.counter4 > HEIGHT_BAR1_THRESHOLD:
                    return False
                else:
                    return True
            else:
                return False
        else:
            return True

    def on_button1_click(self, event):
        """Makes the tubes fills up or down on a click

        :param event:
        :return:
        """
        if self.tube1_adv and self.counter1 >= 200.0:
            self.counter1 -= 1
            self.tube1_adv = False
        else:
            if not self.tube1_adv and self.counter1 <= -1.0:
                self.counter1 += 1
                self.tube1_adv = True
            else:
                if self.tube1_adv:
                    self.tube1_adv = False
                else:
                    self.tube1_adv = True

    def on_button2_click(self, event):
        """Same template as the button_1

        :param event:
        :return:
        """
        if self.tube2_adv and self.counter2 >= 200.0:
            self.counter2 -= 1
            self.tube2_adv = False
        else:
            if not self.tube2_adv and self.counter2 <= -1.0:
                self.counter2 += 1
                self.tube2_adv = True
            else:
                if self.tube2_adv:
                    self.tube2_adv = False
                else:
                    self.tube2_adv = True

    def on_button3_click(self, event):
        if self.tube3_adv and self.counter3 >= 200.0:
            self.counter3 -= 1
            self.tube3_adv = False
        else:
            if not self.tube3_adv and self.counter3 <= -1.0:
                self.counter3 += 1
                self.tube3_adv = True
            else:
                if self.tube3_adv:
                    self.tube3_adv = False
                else:
                    self.tube3_adv = True

    def on_button4_click(self, event):
        if self.tube4_adv and self.counter4 >= 200.0:
            self.counter4 -= 1
            self.tube4_adv = False
        else:
            if not self.tube4_adv and self.counter4 <= -1.0:
                self.counter4 += 1
                self.tube4_adv = True
            else:
                if self.tube4_adv:
                    self.tube4_adv = False
                else:
                    self.tube4_adv = True

    def control_circle(self):
        """Controls the movement of the circle

        :return:
        """
        x = self.circle_sprite.center_x
        y = self.circle_sprite.center_y

        if x == POINT_LEFT_X and y == POINT_UP_Y:
            self.circle_mvt_state = RIGHT
        if x == POINT_RIGHT_X and y == POINT_UP_Y:
            self.circle_mvt_state = DOWN
        if x == POINT_RIGHT_X and y == POINT_DOWN_Y:
            self.circle_mvt_state = LEFT
        if x == POINT_LEFT_X and y == POINT_DOWN_Y:
            self.circle_mvt_state = UP

        if maybe_stop(1.0):  # Stops the circle and begins incrementing the circle score
            self.circle_mvt_state = STOP
        actual_state = self.circle_mvt_state
        if actual_state == RIGHT:
            self.circle_sprite.center_x += 1
            self.remember_previous_state = RIGHT
        if actual_state == DOWN:
            self.circle_sprite.center_y -= 1
            self.remember_previous_state = DOWN
        if actual_state == LEFT:
            self.circle_sprite.center_x -= 1
            self.remember_previous_state = LEFT
        if actual_state == UP:
            self.circle_sprite.center_y += 1
            self.remember_previous_state = UP

    def on_mouse_press(self, x: float, y: float, button: int, modifiers: int):
        """Make the circle continues his course if stopped

        :param x: x position of click
        :param y: y position of click
        :param button: right or left click
        :param modifiers:
        :return:
        """
        try:
            sprite = arcade.get_sprites_at_point((x, y), self.sprite_list)
            if sprite[0] == self.circle_sprite:
                self.circle_mvt_state = self.remember_previous_state
        except IndexError:
            print("No sprite detected")

    def update_text1(self):
        """Update the upper text and the font color of the upper text for the color question

        :return:
        """
        if not self.question_color_on:  # Don't change the question if not already answered
            self.color_chosen = arcadeColors[COLORS[random.randrange(0, len(COLORS) - 1)]]  # Random color font
            index1 = random.randrange(0, len(COLORS) - 1)
            self.label1.text = COLORS[index1]  # random name of color
            self.question_color_on = True
            self.label1.color = self.color_chosen

    def update_text2(self):
        """Update the bottow text to ask simple addition questions

        :return:
        """
        if not self.question_math_on:  # Don't change the question if not already answered
            a = random.randrange(1, 10)
            b = random.randrange(1, 10)  # random sum
            self.label2.text = str(a) + " + " + str(b) + "?"
            self.sum = a + b
            self.question_math_on = True

    def update_button_left(self):
        """Update the color of the upper left light (not working)

        :return:
        """
        coin = random.randint(0, 1)
        if coin:
            self.up_light1.color = arcade.color.GREEN
        else:
            self.up_light1.color = arcade.color.RED

    def update_button_right(self):
        """Update the color of the upper right light (not working)

        :return:
        """
        coin = random.randint(0, 1)
        if coin:
            self.up_light2.color = arcade.color.RED
        else:
            self.up_light2.color = arcade.color.GREEN

    def on_click_button_left(self, event):
        """Changes the color back to black if green ( not working )

        :param event:
        :return:
        """
        if self.up_light1.color == arcade.color.GREEN:
            self.up_light1.color = arcade.color.BLACK
        if self.up_light2.color == arcade.color.GREEN:
            self.up_light2.color = arcade.color.BLACK

    def on_click_button_right(self, event):
        """Changes the color back to black if red ( not working )

        :param event:
        :return:
        """
        if self.up_light1.color == arcade.color.RED:
            self.up_light1.color = arcade.color.BLACK
        if self.up_light2.color == arcade.color.RED:
            self.up_light2.color = arcade.color.BLACK

    def on_key_press(self, symbol: int, modifiers: int):
        """Activates on enter key pressed, check if the text in the input match the answers of the text labels,
        color or math

        :param symbol: key pressed
        :param modifiers:
        :return:
        """
        if symbol == arcade.key.ENTER:
            submitted_text = self.input_field.text
            self.input_field.text = ''
            if submitted_text == str(self.sum):
                self.label2.text = "SUCCESS"
                self.question_math_on = False
            if arcadeColors[submitted_text] == self.color_chosen:
                self.label1.text = "SUCCESS"
                self.question_color_on = False
                self.label1.color = arcade.color.YELLOW

    def on_key_release(self, _symbol: int, _modifiers: int):
        """Empty the input text on release of enter

        :param _symbol: key released
        :param _modifiers:
        :return:
        """
        if _symbol == arcade.key.ENTER:
            self.input_field.text = ''

    def end_game(self, event=None):
        """End the game and shows the game over view

        :param event:
        :return:
        """
        if event is not None:  # in case of game interruption before the 10 minutes timer
            self.clear()
            text = "Game ended, your score has not been saved"
            alert = arcade.gui.UIMessageBox(width=500, height=50, message_text=text, callback=self.close_game,
                                            buttons=["Ok"])
            self.ui_manager.add(alert)

        else:
            self.clear()
            self.ui_manager.clear()
            scoreSaving.scoreWrite(self.player_first_name, self.player_last_name, self.score, self.level)
            self.window.show_view(
                GameOverView.GameOverView(self.player_first_name, self.player_last_name, self.score, self.level))

    def close_game(self, event):
        """Closes the game

        :param event:
        :return:
        """
        self.window.close()


if __name__ == '__main__':
    pass
