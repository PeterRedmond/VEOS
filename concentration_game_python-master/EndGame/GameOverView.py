import arcade
from arcade import gui
import ctypes
user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
SCREEN_WIDTH = screensize[0]
SCREEN_HEIGHT = screensize[1]

"""
Game over view, display final score
"""
class GameOverView(arcade.View):
    def __init__(self, first_name, last_name, score, level):
        super().__init__()

        """
        Check level for score display
        """
        if level > 1:
            self.everything_at_once = True
        else:
            self.everything_at_once = False

        """
        Ui manager creation
        """
        self.ui_manager = arcade.gui.UIManager()
        self.ui_manager.enable()

        """
        Vertical box creation
        """
        self.v_box1 = arcade.gui.UIBoxLayout(space_between=25)

        """
        Label for end game welcoming
        """
        self.label_game_over = arcade.gui.UILabel(text="Game Over\n\rThis is your score " + first_name + " " + last_name, align="center")

        """
        Second vertical box creation to be added to the first
        """
        self.v_box2 = arcade.gui.UIBoxLayout(space_between=5)

        """
        Display the different scores in the last vertical box
        """
        tubes_score = score["tubes"]
        self.score1 = arcade.gui.UILabel(text=f"Tube Score : {tubes_score:.2f}")
        self.v_box2.add(self.score1)
        if self.everything_at_once: #if level not one
            circle_score = score["circle"]
            self.score2 = arcade.gui.UILabel(text=f"Circle Score : {circle_score:.2f}")
            questions_score = score["questions"]
            self.score3 = arcade.gui.UILabel(text=f"Questions Score : {questions_score:.2f}")
            lights_score = score["lights"]
            self.score4 = arcade.gui.UILabel(text=f"Lights Score : {lights_score:.2f}")
            self.v_box2.add(self.score2)
            self.v_box2.add(self.score3)
            self.v_box2.add(self.score4)

        """
        Adding the second vertical box and the label to the first vertical box
        """
        self.v_box1.add(self.label_game_over)
        self.v_box1.add(self.v_box2)

        """
        Button to close the window
        """
        self.button_end = arcade.gui.UIFlatButton(width=200, height=50, text="Close game")
        self.button_end.on_click = self.close_window
        self.v_box1.add(self.button_end)
        self.ui_manager.add(
            arcade.gui.UIAnchorWidget(
                child=self.v_box1
            )
        )

    def on_draw(self):
        self.clear()
        self.ui_manager.draw()

    def close_window(self, event):
        self.window.close()
