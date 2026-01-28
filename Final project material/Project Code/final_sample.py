# Face Recognition 
import face_recognition as fr
import cv2
import time
from datetime import datetime as dt
import numpy as np
import os
import mysql.connector
import pickle
import pandas as pd
import tkinter as tk					
from tkinter import *
from tkinter import ttk
import mysql.connector
from functools import partial
from PIL import ImageTk, Image



conn = mysql.connector.connect( host = 'localhost', 
                                user = 'root',
                                password = '',
                                database = 'imagedb')
cursor = conn.cursor()

def show(cv2_op:Label, windows):
    known_face_encodings = []
    local_enc= []
    cursor = conn.cursor()
    cursor.execute('select `face_encodings` from `known`')
    rows = cursor.fetchall()
    if len(rows) != 0:
        for each in rows:
            for data in each:
                fetched_data = pickle.loads(data)
                # print(np.array(fetched_data))
                known_face_encodings.append(fetched_data)               #from db for recognition
                local_enc.append(fetched_data)
    # print(known_face_encodings)
    # known_local_enc = known_face_encodings                                    #for local unkown face info

    known_local_names = []
    cursor.execute('select `name` from `known`')
    row = cursor.fetchall()
    if len(row) != 0:
        for each in row:
            for i in each:
                known_local_names.append(i)                              #from db of known face names for recognition

    unknown_local_enc = []
    cursor.execute('select `face_encodings` from `unknown`')
    rowss = cursor.fetchall()
    if len(rowss) != 0:
        for each in rowss:
            for data in each:
                fetched_data = pickle.loads(data)
                unknown_local_enc.append(fetched_data)
                local_enc.append(fetched_data)


    static_back = None
    motion_list = [ None, None ]
    time = []
    df = pd.DataFrame(columns = ["Start", "End"])
    video = cv2.VideoCapture(0)

    cascpath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascpath)
    while True:
        ret, frame = video.read()
        motion = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)	
        
        if static_back is None:
            static_back = gray
            continue

        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
        cnts,_ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 10050:
                continue
            motion = 1

            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        motion_list.append(motion)
        motion_list = motion_list[-2:]
        
        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time.append(dt.now())
        
        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time.append(dt.now())


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        current_enc = []
        faces1 = frame[:, :, ::-1]
        face_locations = fr.face_locations(faces1)
        face_encodings = fr.face_encodings(faces1, face_locations)


        for face_encs in face_encodings:
            current_enc.append(face_encs)

        count  = -1
        for (x, y, w, h), face_enc in zip(faces, current_enc):

            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            faces = frame[y:y + h, x:x + w]

            face_enc_dump = pickle.dumps(face_enc)

            flag = 0


            x1 = dt.now()
            date = x1.strftime('%y')
            date += x1.strftime('%m')
            date += x1.strftime('%d')
            date += x1.strftime('%H')
            date += x1.strftime('%M')
            date += x1.strftime('%S')
            date += str(count)
            name = "img" + date

            if len(local_enc) == 0:
                query = f"insert into `unknown` (`face_encodings`, `img_name`) values(%s, '{name}')"
                cursor.execute(query, (face_enc_dump,))
                conn.commit()
                faces = frame[y: y + h, x: x + w]
                cv2.imwrite(f'storage/unknown/{name}.jpg', faces)
                local_enc.append(face_enc)
                flag = 1
            else:
                matches = fr.compare_faces(local_enc, face_enc)
                fd = fr.face_distance(local_enc, face_enc)
                best_index = np.argmin(fd)
                # print(matches)
                if matches[best_index]:
                    # print('first')
                    flag = 1
                    # break

            if flag != 1:
                # print('here')
                # print(xnn)
                query = f"insert into `unknown` (`face_encodings`, `img_name`) values(%s, '{name}')"
                cursor.execute(query, (face_enc_dump,))
                conn.commit()
                faces = frame[y: y + h, x: x + w]
                cv2.imwrite(f'storage/unknown/{name}.jpg', faces)
                local_enc.append(face_enc)
                

                
            
        # rgb_frame = frame[:, :, ::-1]

        # face_locations = fr.face_locations(rgb_frame)
        # face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # print(len(known_face_encodings))
            if len(known_face_encodings) == 0:
                name = "Unknown"

                cv2.rectangle(frame, (left + 10, top), (right + 10, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left + 10, bottom - 40), (right + 10, bottom + 20), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 20, bottom + 6), font, 1.0, (255, 255, 255), 1)

            else:            
                
                matches = fr.compare_faces(known_face_encodings, face_encoding)

                name = "Unknown"
                # print(matches)
                # print(x)
                face_distances = fr.face_distance(known_face_encodings, face_encoding)

                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = known_local_names[best_match_index]

                cv2.rectangle(frame, (left + 10, top), (right + 10, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left + 10, bottom - 40), (right + 10, bottom + 20), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, name, (left + 20, bottom + 6), font, 1.0, (255, 255, 255), 1)
                    # i += 1
        
        # cv2.imshow("Color Frame", frame)
        # # capture = cv2.cvtColor()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
        cv2_op['image'] = img
        

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        else:
            windows.update()

    for i in range(0, len(time), 2):
        df = pd.concat([df, pd.DataFrame.from_records([{"Start":time[i], "End":time[i + 1]}])])

    df.to_csv("Time_of_movements.csv")


    video.release()
    cv2.destroyAllWindows()
    # cursor.close()
    # conn.close()



def search(cursor, result_label:Label, simage:Label, winn2:Frame):
    winn2.pack(side = 'bottom', fill = 'both')
    name = search_name.get()
    if name == '':
        result_label.set('Please enter valid name')
    else:
        result_label.set('')
        cursor.execute(f"select `name` from `known` where `name` = '{name}'")
        result = cursor.fetchall()
        if len(result) == 0:
            pname = 'Null'
            # result_label.configure(text = f'Sorry, no data available of name: {search_name.get()}')
            result_label.set(f'Sorry, no data available of name: {search_name.get()}')
        else:
            pname = result[0][0]
            paths = f'storage/known/{pname}.jpg'
            img = ImageTk.PhotoImage(Image.open(paths))
            simage.configure(image = img)
            simage.image = img

def prev(path, i, panel:Label):

    n = len(path)
    if i == 0:
        i = n -1
        img = Image.open(path[i])
        test = ImageTk.PhotoImage(img)
        panel.configure(image = test)
        panel.image = test
        i_var.set(i)
    else:
        i = i - 1
        test = ImageTk.PhotoImage(Image.open(path[i]))
        panel.configure(image = test)
        panel.image = test
        i_var.set(i)
        
def next(path, i, panel:Label):

    n = len(path)
    if i == n - 1:
        i = 0
        test = ImageTk.PhotoImage(Image.open(path[i]))
        panel.configure(image = test)
        panel.image = test
        i_var.set(i)
    else:
        i = i + 1
        test = ImageTk.PhotoImage(Image.open(path[i]))
        panel.configure(image = test)
        panel.image = test
        i_var.set(i)

def register(conn, cursor, image_name, face_enc, i, path):
    n = len(path)
    name = str(name_var.get())
    if name == '':
        rmsg_error.set('Please enter a valid name')
    else:
        cursor.execute(f"select * from `known` where `name`= '{name}'")
        res = cursor.fetchall()
        if len(res) == 0:
            rmsg_error.set('')
            query = f"insert into `known` (`name`, `face_encodings`) values('{name}',%s)"
            cursor.execute(query, (face_enc[i],))
            conn.commit()
            source = f'storage/unknown/{image_name[i]}.jpg'
            dest = f'storage/known/{name}.jpg'
            os.rename(source, dest)
            query = f'delete from `unknown` where `img_name` = "{image_name[i]}"' 
            cursor.execute(query)
            conn.commit()        
            del(image_name[i])
            del(path[i])
            del(face_enc[i])
            i -= 1
            i_var.set(i)
            n -= 1
            rmsg_error.set('Name registered successfully...!')
            name_var.set('')
        else:
            rmsg_error.set('Name is already present in database. Please enter their name with surname or nickname.')

def update(conn, cursor):
    pname = search_name.get()
    nname1 = nname.get()

    if nname1 == '':
        msg_var.set('Please enter valid name')
    else:
        cursor.execute(f"select * from `known` where `name` = '{nname1}'")
        res = cursor.fetchall()
        if len(res) == 0:
            msg_var.set('')
            cursor.execute(f"update `known` set `name` = '{nname1}' where `name` = '{pname}'")
            conn.commit()
            msg_var.set('Name updated successfully...!')
            spath = f'storage/known/{pname}.jpg'
            dpath = f'storage/known/{nname1}.jpg'
            os.replace(spath, dpath)
            nname.set('')
            search_name.set('')
        else:
            msg_var.set('Name is already present in database. \nPlease enter their name with surname or nickname.')

# conn = mysql.connector.connect(host = 'localhost', user = 'root', password = '', database = 'imagedb')

# cursor = conn.cursor()

path =[]
face_enc = []
image_name = []
images = []


           

windows = tk.Tk()
width = windows.winfo_screenwidth()               
height = windows.winfo_screenheight()               
windows.geometry("%dx%d" % (width, height))
windows.config(background='#59595b')
windows.title("Motion Detection & Multiple Faces Identification")
tabControl = ttk.Notebook(windows, padding = 20)

name_var = tk.StringVar()
return_var = tk.IntVar()
i_var = tk.IntVar()
search_name = tk.StringVar()
i_var.set(0)
commit_var = tk.IntVar()
msg_var = tk.StringVar()
nname = tk.StringVar()
result_label = tk.StringVar()
rmsg_error = tk.StringVar()


#Fetch Unknown Database


s = ttk.Style()
s.theme_use('default')
s.configure('TNotebook.Tab', background = '#5962e0', padding = [10, 20], font = ("sans-serif", 15, 'bold'))
s.map('TNotebook', activebackground = "#5962e0")

tab1 = ttk.Frame(tabControl)
tab2 = ttk.Frame(tabControl)
tab3 = ttk.Frame(tabControl)
tab4 = ttk.Frame(tabControl)



tabControl.add(tab1, text = 'About US', padding = 20)
tabControl.add(tab2, text = 'Motion Detection and Face Identification', padding = 20)
tabControl.add(tab3, text = 'Registration', padding = 20)
tabControl.add(tab4, text = 'Update Name', padding = 20)
tabControl.pack(expand = 1, fill = "both")

image1 = Image.open('images/KJ Logo.jpg')
test = ImageTk.PhotoImage(image1)


#About us
ttk.Label(tab1, image = test).grid(row = 0, column = 0, sticky = W)
ttk.Label(tab1, text = "\t\t\t\t").grid(row = 0, column = 1, sticky = W)
ttk.Label(tab1, text = "KJ's College of Engineering and Management Research, Pisoli\n \tDepartment of Computer Engineering", font=('sans-serif', 20, 'bold')).grid(row = 0, column = 2)
ttk.Separator(tab1, orient = 'horizontal').grid(row = 1, column = 0, columnspan = 99, ipadx = 1000, pady = 10, sticky = W)
ttk.Label(tab1, text = '\t\t\t\t').grid(row = 2, column = 0, sticky = W)
ttk.Label(tab1, text = 'About Us', font=('sans-serif', 18, 'bold')).grid(row = 2, column = 1, columnspan = 2, padx = 10, pady = 10)
ttk.Label(tab1, text = '\t\t\t\t').grid(row = 3, column = 0, sticky = W)
ttk.Label(tab1, text = '''\tAs technology is marking the highest peak in this modern world, all things are being automated using Machine Learning and Deep Learning. By using Machine Learning models, we can detect the movements, action, multiple faces, their recognition and many more in just a second of time. '''
'''\n\n\tThe title “Motion Detection and Multiple Faces Identification using Webcam” tells us much more about this system. The “Motion Detection” were the actions made by an object is going to be observed. And the “Multiple Faces Identification” were detection and recognition of multiple faces is done at same time. Both model use Webcam / Security Camera for detecting motion and multiple faces in single video frame, and a database is used to recognize multiple faces and storing of newly detected face. 
''', font = ('sans-serif', 16), wraplength = 1100, justify = LEFT).grid(row = 3, column = 1, columnspan = 3, sticky = W, pady = 50)


#Motion Detection & Face Identification
ttk.Label(tab2, text ="\t\t\t\tMotion Detection & Multiple Face Identification", font = ('Sans-serif', 20, 'bold')).pack(side = 'top', fill = 'both')
ttk.Label(tab2, text = '\t\t\t').pack(side = 'top', fill = 'x')
ttk.Label(tab2, text = '\t\t\t\t\t\t\t\t\t').pack(side = 'left')
cv2_op = ttk.Label(tab2, font = ('sans-serif', 18, 'bold'))
cv2_op.pack(side = 'top', fill = 'both')


#Registration
def reset(conn, cursor, path):
    cursor.execute('select `face_encodings`, `img_name` from unknown')
    result = cursor.fetchall()

    if len(result) == 0:
        ttk.Label(tab3, text = "Sorry, there's no data to display...!", font = ('sans-serif', 20, 'bold'), padding = 20).pack(side = 'top')

    else:
        for i in result:
            for p in i:
                if isinstance(p, str) == True:
                    path.append(f'storage/unknown/{p}.jpg')
                    image_name.append(p)

                else:
                    face_enc.append(p)

        n = len(path)


        win1 = Frame(tab3, height = 700, width = 800, background = '#d9d9d9')
        ttk.Label(win1, text = 'Registration', font = ('sans-serif', 20, 'bold'), padding = 20).pack(side = 'top')

        tt = ImageTk.PhotoImage(Image.open(path[i_var.get()]))
        panel = tk.Label(win1, image = tt)
        panel.image = tt
        panel.pack(side = 'top')

        ttk.Label(win1, text = '\t\t\t\t\t', padding = 20).pack(side = 'top')
        ttk.Label(win1, text = 'Enter Name: ', padding = 20, font = ('sans-serif', 17, 'bold')).pack(side = 'top')
        name1 = ttk.Entry(win1, textvariable = name_var, font = ('sans-serif', 15), background = '#ffffff').pack(side = 'top')

        ttk.Label(win1, text = '\t\t\t\t', textvariable = rmsg_error, font = ('sans-serif', 16, 'bold'), padding = 20).pack(side = 'top')
        ttk.Label(win1, text = '\t\t\t\t', padding = 20).pack(side = 'top')

        ttk.Button(win1, text = 'Submit', padding = 20, command = lambda: register(conn, cursor, image_name, face_enc, i_var.get(), path)).pack(side = 'top')
        
        ttk.Button(tab3, text = '<- Previous', command = lambda: prev(path, i_var.get(), panel), padding = 20).pack(side = 'left')
        win1.pack(side = 'left', expand = True)
        ttk.Button(tab3, text = 'Next ->', padding = 20, command = lambda: next(path, i_var.get(), panel)).pack(side = 'left')


#Update Name
def upreset(conn, cursor):
    winn1 = Frame(tab4, width = 800, height = 200, background = '#ffffff')
    winn2 = Frame(tab4, width = 800, height = 700, background = '#fffffd')


    ttk.Label(winn1, text = 'Enter name of person to search: ', font = ('sans-serif', 16, 'bold'), padding = 20, background = '#ffffff').pack(side = 'left', fill = 'both')
    sname = ttk.Entry(winn1, text ='', textvariable= search_name, font = ('sans-serif', 16), background = '#ffffff').pack(side = 'left')
    ttk.Label(winn1, text = '\t\t', padding = 20, background = '#ffffff').pack(side = 'left')
    simage = ttk.Label(winn2, padding = 20, background = '#ffffff')
    ttk.Button(winn1, text = 'Search', padding = 10, command = lambda: search(cursor, result_label, simage, winn2)).pack(side = 'left')
    ttk.Label(winn2, textvariable = result_label, font = ('sans-serif', 20, 'bold'), background = '#ffffff', padding = 20).place(x = 500, y = 50)

    simage.place(x = 620, y = 100)
    ttk.Label(winn2, text = 'Enter a new name:', font = ('sans-serif', 18, 'bold'), background = '#ffffff', padding = 20).place(x = 600, y = 330)
    nnname = ttk.Entry(winn2, textvariable = nname, font = ('sans-serif', 16), background = '#ffffff')
    nnname.place(x = 600, y = 400)

    msg = ttk.Label(winn2, text = '', textvariable = msg_var, font = ('sans-serif', 18, 'bold'), background = '#ffffff', padding = 20)
    msg.place(x = 570, y = 450)
    # ttk.Label(winn2, text = '\t').place(x = 570, y = )
    ttk.Button(winn2, text = 'Submit', padding = 20, command = lambda: update(conn, cursor)).place(x = 670, y = 550)
    winn1.pack(side = 'top', fill = 'both')

# show = partial(show, cv2_op, windows)

def show1(event):
    seletced_tab = event.widget.select()
    tab_text = event.widget.tab(seletced_tab, "text")

    if tab_text == 'Motion Detection and Face Identification':
        show(cv2_op, windows)
    
    if tab_text == 'Registration':
        reset(conn, cursor, path)

    if tab_text == 'Update Name':
        upreset(conn, cursor)
    

tabControl.bind('<<NotebookTabChanged>>', show1)
# tab3.bind('<<Notebook')

windows.mainloop()
