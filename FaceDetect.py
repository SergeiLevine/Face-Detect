#!/usr/bin/python

import tkinter, Tkconstants, tkFileDialog, tkMessageBox
from tkinter import *
global filedir
def getdirectory():
    global filedir
    filedir = tkFileDialog.askdirectory()
    tkMessageBox.showinfo("The directory", filedir)
def pri():
    global filedir
    print(filedir)

root = Tk()
var = StringVar()


label = Label( root, textvariable=var, relief=RAISED )
button = Button(root, text ="Browse", command = lambda: getdirectory())
button1 = Button(root, text ="Browse", command = lambda: pri())
var.set("Pleace enter path")

label.pack()
button.pack()
button1.pack()




root.minsize(width=500, height=500)
mainloop()

