import arcade
import arcade.gui
from Game import GameView

class LevelChoice(arcade.View):
    def __init__(self, first_name, last_name):
        super().__init__()

        """
        Stocks the informations from info input        
        """
        self.player_first_name = first_name
        self.player_last_name = last_name

        """
        Ui manager creation
        """
        self.ui_manager = arcade.gui.UIManager()
        self.ui_manager.enable()

        """
        Vertical box creation
        """
        self.v_box = arcade.gui.UIBoxLayout(space_between=100)

        """
        Welcome message label
        """
        self.label_player = arcade.gui.UILabel(text="Welcome " + self.player_first_name + " " + self.player_last_name)
        self.v_box.add(self.label_player)

        """
        Horizontal box creation, stocks the three buttons for level selection
        """
        self.h_box = arcade.gui.UIBoxLayout(vertical=False, space_between=150)
        self.button1 = arcade.gui.UIFlatButton(text="Level 1")
        self.button2 = arcade.gui.UIFlatButton(text="Level 2")
        self.button3 = arcade.gui.UIFlatButton(text="Level 3")

        """
        Functions assignements to functions
        """
        self.button1.on_click = self.on_click1
        self.button2.on_click = self.on_click2
        self.button3.on_click = self.on_click3

        """
        Addings buttons to the horizontal box
        """
        self.h_box.add(self.button1)
        self.h_box.add(self.button2)
        self.h_box.add(self.button3)

        """
        Adding the horizontal box inside the vertical box
        """
        self.v_box.add(self.h_box)

        """
        Add the vertical box inside an anchor widget to display it on the view
        """
        self.ui_manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="center_x",
                anchor_y="center_y",
                child=self.v_box
            )
        )

    def on_draw(self): #Draw background and widgets
        self.clear()
        arcade.set_background_color(arcade.color.BLACK)
        self.ui_manager.draw()

    def on_click1(self, event):
        """Starts the game on level 1

        :param event:  not used
        :return:
        """
        self.window.show_view(GameView.GameView(1, self.player_first_name, self.player_last_name))
        self.ui_manager.clear()
        self.clear()

    def on_click2(self, event):
        """Starts the game on level 2

        :param event:
        :return:
        """
        self.window.show_view(GameView.GameView(2, self.player_first_name, self.player_last_name))
        self.ui_manager.clear()
        self.clear()

    def on_click3(self, event):
        """Starts the game on level 3

        :param event:
        :return:
        """
        self.window.show_view(GameView.GameView(3, self.player_first_name, self.player_last_name))
        self.ui_manager.clear()
        self.clear()
