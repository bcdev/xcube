from chartlets import Extension
from .my_panel_1 import panel as my_panel_1
from .my_panel_2 import panel as my_panel_2
from .my_panel_3 import panel as my_panel_3

ext = Extension(__name__)
ext.add(my_panel_1)
ext.add(my_panel_2)
ext.add(my_panel_3)
