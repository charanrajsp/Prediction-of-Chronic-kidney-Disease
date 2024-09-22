from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
import sqlite3
import warnings
warnings.filterwarnings("ignore")
#Specific gravity= sg
#sugar=su
#albumin=al

def main():
    R1=Tk()
    R1.geometry('900x800')
    R1.title('Main')
    '''
    image = PIL.Image.open('images.jpg')
    image=image.resize((900,600))
    photo_image = PIL.ImageTk.PhotoImage(image)
    label=Label(R1, image = photo_image).pack()
    #label.place(x=0,y=0)
    '''  
    
    la=Label(R1,text="Chronic kindny prediction",font=('Book Antiqua',20,'bold'))
    la.place(x=200,y=100)
    
    Registerbt = Button(R1,text = "REGISTER",width=17,height=2,font=('Cambria',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=sigup)
    loginbt = Button(R1,text = "LOGIN",width=17,height=2,font=('Cambria',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=login)
    Registerbt.place(x =300 ,y=200)
    loginbt.place( x =300,y=300)
    R1.mainloop()
    

def sigup():
    
    def Sigups():
        
        usernames = username.get()
        passwords = password.get()
        phonenos = phoneno.get()
        Address = address.get()
        conn = sqlite3.connect('ckp.db')
        with conn:
            cursor=conn.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS patient (Username TEXT,Password TEXT,Phoneno TEXT,Address TEXT)')
        cursor.execute('INSERT INTO patient (Username,Password ,Phoneno,Address) VALUES(?,?,?,?)',
                  (usernames,passwords,phonenos,Address,))
        conn.commit()
        if username.get() == "" or password.get() == "":
           messagebox.showinfo("sorry", "Pease fill the required information")
        else:
           messagebox.showinfo("Welcome to %s" % usernames, "Let Login")
           login()
           R2.destroy()
           
           
      
           
    
    R2=Toplevel()
    R2.geometry('900x600')
    R2.title('SigUp Now')
   
   
    lblInfo=Label(R2,text="Username",fg="black",font=('Cambria',"bold",15))
    lblInfo.place(x=200,y=140)

    lblInfo=Label(R2,text="Password",fg="black",font=("bold",15))
    lblInfo.place(x=200,y=190)

    lblInfo=Label(R2,text="phoneno",fg="black",font=("bold",15))
    lblInfo.place(x=200,y=240)

    lblInfo=Label(R2,text="Adress",fg="black",font=("bold",15))
    lblInfo.place(x=200,y=290)

    

    username=Entry(R2,width=20,font=("bold",15),highlightthickness=2)
    username.place(x=300,y= 140 )
    
    password=Entry(R2,show="**",width=20,font=("bold",15),highlightthickness=2)
    password.place(x=300,y=190 )
    
    phoneno=Entry(R2,width=20,font=("bold",15),highlightthickness=2)
    phoneno.place(x=300,y= 240 )
    
    address=Entry(R2,width=20,font=("bold",15),highlightthickness=2)
    address.place(x=300,y= 290 )

    

    signUpbt = Button(R2,text = "SignUp",width=10,height=2,fg="black",font=('algerian',15,'bold'),justify='center',bg="light blue",command=Sigups)
    signUpbt.place( x =350,y=490)
    
      
    R2.mainloop()



def login():
    def logininto():
        
        usernames = e1.get()
        passwords = e2.get()
        conn = sqlite3.connect('ckp.db')
        with conn:
          cursor=conn.cursor()
        ('CREATE TABLE IF NOT EXISTS patient (Username TEXT,Password TEXT,Phoneno TEXT)')
        conn.commit()
        if usernames == "" and passwords == "" :
            messagebox.showinfo("sorry", "Please complete the required field")
        else:
            cursor.execute('SELECT * FROM patient WHERE Username = "%s" and Password = "%s"'%(usernames,passwords))
            if cursor.fetchall():
                messagebox.showinfo("Welcome %s" % usernames, "Logged in successfully")
                R3.destroy()
                main1()
            else:
                messagebox.showinfo("Sorry", "Wrong Password")
  
    
    R3 =Toplevel()
    R3.geometry('900x600')
    R3.title("LOGIN NOW")
    
   
    

    lblInfo=Label(R3,text="username",fg="black",font=("bold",15))
    lblInfo.place(x=230,y=200)
   
    lblInfo=Label(R3,text="Password",fg="black",font=("bold",15))
    lblInfo.place(x=230,y=250)

    e1= Entry(R3,width=15,font=("bold",17),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e1.place(x=330, y=190)

    e2= Entry(R3,width=15,font=("bold",17),show="*",highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e2.place(x=330, y=240)

    btn = Button(R3, text="LOGIN", width=10, height=2,fg="black",font=('algerian',15,'bold'),justify='center',bg="light blue",command=logininto)
    btn.place(x=380, y=400)
    
    R3.mainloop()

def main1():
    def call1():
        R4.destroy()
        prediction()
    R4=Tk()
    R4.geometry('900x600')
    R4.title('Algorithm')
    R4.resizable(width = FALSE ,height= FALSE)
    '''Image_open = Image.open("emo1.jpg")
    image = ImageTk.PhotoImage(Image_open)
    sigup = Label(R2,image=image)
    sigup.place(x=0,y=0,bordermode="outside")'''
   
    Registerbt = Button(R4,text = "PREDICTION",width=17,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",command=prediction)
    
    Registerbt.place(x =300 ,y=200)
    call1()
    R4.mainloop()



def prediction():
    R5=Tk()
    R5.geometry('900x600')
    R5.title('Algorithm')
    R5.resizable(width = FALSE ,height= FALSE)



#----------------svm-----------------
    def svm():
        R5.destroy()
        def svm1():
            
            
            T = E1.get()
            V = E2.get()
            print(type(V))
            H = E3.get()
            PM  = E4.get()
            pp  = E5.get()
            #filename='Pickle_model.pkl'
            #loaded_model = pickle.load(open(filename, 'rb'))
            #result = classifier.predict([[turbidity,tds,temp,ph]])
            result = model.predict([[T,V,H,PM,pp]])
            print(result)
            message.configure(text= acc)
            message1.configure(text=result)
        print('=====================SVM===================')
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Importing the dataset
        dataset = pd.read_csv('k.csv')
        X = dataset.iloc[:, [0,1,2,3,4]].values
        y = dataset.iloc[:, 5].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        # Feature Scaling
        '''from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)'''

        # Fitting K-NN to the Training set
        from sklearn.svm import SVC
        model = SVC()
        model.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = model.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        #z = model.predict([[48,80,1.02,1,0]])
        #print(z)
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(y_test, y_pred)
        print(acc)
           
        #%matplotlib inline
        plt.xlabel('parameters')
        print(0)
        plt.ylabel('target')
        
        window2 =Toplevel()
        window2.geometry('900x800')
        window2.title("LOGIN NOW")
        lb1 = Label(window2, text="Chronic kidny prediction",font=('algerian',15,'bold'),justify='center',fg="BLUE")
        lb1.place(x=40, y=70)
        lb1 = Label(window2, text="Age",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb1.place(x=100, y=120)
        E1=IntVar()
        e1= Entry(window2,width=10,textvariable=E1,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e1.place(x=300, y=120)

            
        lb2 = Label(window2, text="Bp",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb2.place(x=100, y=200)
        E2=DoubleVar()
        e2= Entry(window2,width=10,textvariable=E2,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e2.place(x=300, y=200)

        lb3 = Label(window2, text="Specific gravity",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb3.place(x=100, y=270)
        E3=IntVar()
        e3= Entry(window2,width=10,textvariable=E3,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e3.place(x=300, y=270) 

        lb4 = Label(window2, text="Albumin",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb4.place(x=100, y=350)
        E4=IntVar()
        e4= Entry(window2,width=10,textvariable=E4,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e4.place(x=300, y=350)

        lb5 = Label(window2, text="Sugar",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb5.place(x=100, y=430)
        E5=IntVar()
        e5= Entry(window2,width=10,textvariable=E5,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e5.place(x=300, y=430)
        btn1 = Button(window2, text= "PREDICTION", width=15, height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=svm1)
        btn1.place(x=230, y=510)
        
        lblIn=Label(window2,text="Accuracy",fg="black",font=("bold",15))
        lblIn.place(x=100,y=550)

        lblIn=Label(window2,text="Status",fg="black",font=("bold",15))
        lblIn.place(x=470,y=550)
        
        message =Label(window2, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message.place(x=40, y=590)
        
        message1 =Label(window2, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message1.place(x=350, y=590)
        bt = Button(window2,text = "<--",width=10,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=main)
        bt.place(x =0 ,y=0)
        #message1.configure(text= res)
        window2.mainloop()





#-----------------------random forest----------------------------
    def randomforest():
        R5.destroy()
        print('====================Randomforest===================')
        def rd1():
            
            
            T = E1.get()
            V = E2.get()
            print(type(V))
            H = E3.get()
            PM  = E4.get()
            pp  = E5.get()
            #filename='Pickle_model.pkl'
            #loaded_model = pickle.load(open(filename, 'rb'))
            #result = classifier.predict([[turbidity,tds,temp,ph]])
            result = model.predict([[T,V,H,PM,pp]])
            print(result)
            message.configure(text= acc)
            message1.configure(text=result)
        print('=====================SVM===================')
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Importing the dataset
        dataset = pd.read_csv('k.csv')
        X = dataset.iloc[:, [0,1,2,3,4]].values
        y = dataset.iloc[:, 5].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        # Feature Scaling
        '''from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)'''

        # Fitting K-NN to the Training set
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = model.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        #z = model.predict([[48,80,1.02,1,0]])
        #print(z)
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(y_test, y_pred)
        print(acc)
           
        #%matplotlib inline
        plt.xlabel('parameters')
        print(0)
        plt.ylabel('target')
        
        window1 =Toplevel()
        window1.geometry('900x800')
        window1.title("LOGIN NOW")
        lb1 = Label(window1, text="AIR POLLATION PREDICTION USING MACHINE LEARNING TECHNIC",font=('algerian',15,'bold'),justify='center',fg="BLUE")
        lb1.place(x=40, y=70)
        lb1 = Label(window1, text="Age",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb1.place(x=100, y=120)
        E1=IntVar()
        e1= Entry(window1,width=10,textvariable=E1,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e1.place(x=300, y=120)

            
        lb2 = Label(window1, text="Bp",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb2.place(x=100, y=200)
        E2=DoubleVar()
        e2= Entry(window1,width=10,textvariable=E2,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e2.place(x=300, y=200)

        lb3 = Label(window1, text="Specific gravity",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb3.place(x=100, y=270)
        E3=IntVar()
        e3= Entry(window1,width=10,textvariable=E3,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e3.place(x=300, y=270) 

        lb4 = Label(window1, text="Albumin",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb4.place(x=100, y=350)
        E4=IntVar()
        e4= Entry(window1,width=10,textvariable=E4,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e4.place(x=300, y=350)

        lb5 = Label(window1, text="Sugar",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb5.place(x=100, y=430)
        E5=IntVar()
        e5= Entry(window1,width=10,textvariable=E5,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e5.place(x=300, y=430)
        btn1 = Button(window1, text= "PREDICTION", width=15, height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=rd1)
        btn1.place(x=230, y=510)
        
        lblIn=Label(window1,text="Accuracy",fg="black",font=("bold",15))
        lblIn.place(x=100,y=550)

        lblIn=Label(window1,text="Status",fg="black",font=("bold",15))
        lblIn.place(x=470,y=550)
        
        message =Label(window1, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message.place(x=40, y=590)
        
        message1 =Label(window1, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message1.place(x=350, y=590)
        bt = Button(window1,text = "<--",width=10,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=main)
        bt.place(x =0 ,y=0)
        #message1.configure(text= res)
        window1.mainloop()

       

        


#-----------------------decission tree----------------------------

    def decissiontree():
        R5.destroy()
        print('====================Decissiontree===================')
        
        def dt1():
            
            
            T = E1.get()
            V = E2.get()
            print(type(V))
            H = E3.get()
            PM  = E4.get()
            pp  = E5.get()
            #filename='Pickle_model.pkl'
            #loaded_model = pickle.load(open(filename, 'rb'))
            #result = classifier.predict([[turbidity,tds,temp,ph]])
            result = model.predict([[T,V,H,PM,pp]])
            print(result)
            message.configure(text= acc)
            message1.configure(text=result)
        print('=====================SVM===================')
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Importing the dataset
        dataset = pd.read_csv('k.csv')
        X = dataset.iloc[:, [0,1,2,3,4]].values
        y = dataset.iloc[:, 5].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        # Feature Scaling
        '''from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)'''

        # Fitting K-NN to the Training set
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = model.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        #z = model.predict([[48,80,1.02,1,0]])
        #print(z)
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(y_test, y_pred)
        print(acc)
           
        #%matplotlib inline
        plt.xlabel('parameters')
        print(0)
        plt.ylabel('target')
        
        window3 =Toplevel()
        window3.geometry('900x800')
        window3.title("LOGIN NOW")
        lb1 = Label(window3, text="AIR POLLATION PREDICTION USING MACHINE LEARNING TECHNIC",font=('algerian',15,'bold'),justify='center',fg="BLUE")
        lb1.place(x=40, y=70)
        lb1 = Label(window3, text="Age",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb1.place(x=100, y=120)
        E1=IntVar()
        e1= Entry(window3,width=10,textvariable=E1,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e1.place(x=300, y=120)

            
        lb2 = Label(window3, text="Bp",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb2.place(x=100, y=200)
        E2=DoubleVar()
        e2= Entry(window3,width=10,textvariable=E2,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e2.place(x=300, y=200)

        lb3 = Label(window3, text="Specific gravity",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb3.place(x=100, y=270)
        E3=IntVar()
        e3= Entry(window3,width=10,textvariable=E3,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e3.place(x=300, y=270) 

        lb4 = Label(window3, text="Albumin",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb4.place(x=100, y=350)
        E4=IntVar()
        e4= Entry(window3,width=10,textvariable=E4,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e4.place(x=300, y=350)

        lb5 = Label(window3, text="Sugar",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb5.place(x=100, y=430)
        E5=IntVar()
        e5= Entry(window3,width=10,textvariable=E5,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e5.place(x=300, y=430)
        btn1 = Button(window3, text= "PREDICTION", width=15, height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=dt1)
        btn1.place(x=230, y=510)
        
        lblIn=Label(window3,text="Accuracy",fg="black",font=("bold",15))
        lblIn.place(x=100,y=550)

        lblIn=Label(window3,text="Status",fg="black",font=("bold",15))
        lblIn.place(x=470,y=550)
        
        message =Label(window3, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message.place(x=40, y=590)
        
        message1 =Label(window3, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message1.place(x=350, y=590)
        bt = Button(window3,text = "<--",width=10,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=main)
        bt.place(x =0 ,y=0)
        #message1.configure(text= res)
        window3.mainloop()
#------------------naive bays----------------------

    def naivebays():
        R5.destroy()
        def nb1():
            
            
            T = E1.get()
            V = E2.get()
            print(type(V))
            H = E3.get()
            PM  = E4.get()
            pp  = E5.get()
            #filename='Pickle_model.pkl'
            #loaded_model = pickle.load(open(filename, 'rb'))
            #result = classifier.predict([[turbidity,tds,temp,ph]])
            result = model.predict([[T,V,H,PM,pp]])
            print(result)
            message.configure(text= acc)
            message1.configure(text=result)
        print('=====================SVM===================')
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Importing the dataset
        dataset = pd.read_csv('k.csv')
        X = dataset.iloc[:, [0,1,2,3,4]].values
        y = dataset.iloc[:, 5].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

        # Feature Scaling
        '''from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)'''

        # Fitting K-NN to the Training set
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = model.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        #z = model.predict([[48,80,1.02,1,0]])
        #print(z)
        from sklearn.metrics import accuracy_score
        acc=accuracy_score(y_test, y_pred)
        print(acc)
           
        #%matplotlib inline
        plt.xlabel('parameters')
        print(0)
        plt.ylabel('target')
        
        window4 =Toplevel()
        window4.geometry('900x800')
        window4.title("LOGIN NOW")
        lb1 = Label(window4, text="AIR POLLUATION PREDICTION USING MACHINE LEARNING TECHNIC",font=('algerian',15,'bold'),justify='center',fg="BLUE")
        lb1.place(x=40, y=70)
        lb1 = Label(window4, text="Age",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb1.place(x=100, y=120)
        E1=IntVar()
        e1= Entry(window4,width=10,textvariable=E1,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e1.place(x=300, y=120)

            
        lb2 = Label(window4, text="Bp",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb2.place(x=100, y=200)
        E2=DoubleVar()
        e2= Entry(window4,width=10,textvariable=E2,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e2.place(x=300, y=200)

        lb3 = Label(window4, text="Specific gravity",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb3.place(x=100, y=270)
        E3=IntVar()
        e3= Entry(window4,width=10,textvariable=E3,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e3.place(x=300, y=270) 

        lb4 = Label(window4, text="Albumin",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb4.place(x=100, y=350)
        E4=IntVar()
        e4= Entry(window4,width=10,textvariable=E4,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e4.place(x=300, y=350)

        lb5 = Label(window4, text="Sugar",font=('algerian',15,'bold'),fg="BLUE",anchor='w')
        lb5.place(x=100, y=430)
        E5=IntVar()
        e5= Entry(window4,width=10,textvariable=E5,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e5.place(x=300, y=430)
        btn1 = Button(window4, text= "PREDICTION", width=15, height=1,fg="black",font=('algerian',15,'bold'),bg="SKYBLUE",justify='center',command=nb1)
        btn1.place(x=230, y=510)
        
        lblIn=Label(window4,text="Accuracy",fg="black",font=("bold",15))
        lblIn.place(x=100,y=550)

        lblIn=Label(window4,text="Status",fg="black",font=("bold",15))
        lblIn.place(x=470,y=550)
        
        message =Label(window4, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message.place(x=40, y=590)
        
        message1 =Label(window4, text="" ,bg="yellow"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message1.place(x=350, y=590)
        bt = Button(window4,text = "Back",width=10,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=main)
        bt.place(x =0 ,y=0)
        #message1.configure(text= res)
        window4.mainloop()        


    b1 = Button(R5,text = "SVM",width=17,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=svm)
    b1.place(x =50 ,y=200)
    b2 = Button(R5,text = "RANDOM FOREST",width=17,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=randomforest)
    b2.place( x =250,y=300)
    b3 = Button(R5,text = "DECISSION TREE",width=17,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=decissiontree)
    b3.place(x =450 ,y=400)
    b4 = Button(R5,text = "NAIVE BAYS",width=17,height=2,font=('algerian',15,'bold'),justify='center',bg="light blue",relief=SUNKEN,command=naivebays)
    b4.place( x =650,y=500)
    
    R5.mainloop()


main()




