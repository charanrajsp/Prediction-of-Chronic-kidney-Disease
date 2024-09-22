from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk
import sqlite3
import warnings
warnings.filterwarnings("ignore")
#Specific gravity= sg
#serum urate=su
#albumin=al

def main():
    R1=Tk()
    R1.geometry('900x500')
    R1.title('Main Page')
    '''
    Image = PIL.Image.open('E:\Download\2017-bentley-mulsanne-32_1600x0.webp')
    Image=Image.resize((900,600))
    photo_Image = PIL.ImageTk.PhotoImage(Image)
    label=Label(R1, Image = photo_Image).pack()
    label.place(x=20,y=40)
    ''' 
    
    la=Label(R1,text="Prediction of Chronic Kidney Disease",justify='center',bg="darkred",fg="white",font=('Book Antiqua',25,'bold',))
    la.place(x=150,y=60)
    la1=Label(R1,text="by (Athreya B.N,Charanraj Pattar)",justify='center',bg="darkred",fg="white",font=('Book Antiqua',15,'bold',))
    la1.place(x=150,y=120)
    Registerbt = Button(R1,text = "REGISTER",width=17,height=2,font=('Book Antiqua',15,'bold'),justify='center',bg="gold",relief=SUNKEN,command=sigup)
    loginbt = Button(R1,text = "LOGIN",width=17,height=2,font=('Book Antiqua',15,'bold'),justify='center',bg="gold",relief=GROOVE,command=login)
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
   
   
    lblInfo=Label(R2,text="Username",fg="black",font=("Book Antiqua bold",15))
    lblInfo.place(x=200,y=140)

    lblInfo=Label(R2,text="Password",fg="black",font=("Book Antiqua bold",15))
    lblInfo.place(x=200,y=190)

    lblInfo=Label(R2,text="phoneno",fg="black",font=("Book Antiqua bold",15))
    lblInfo.place(x=200,y=240)

    lblInfo=Label(R2,text="Adress",fg="black",font=("Book Antiqua bold",15))
    lblInfo.place(x=200,y=290)

    

    username=Entry(R2,width=20,font=("Book Antiqua bold",15),highlightthickness=2)
    username.place(x=300,y= 140 )
    
    password=Entry(R2,show="**",width=20,font=("Book Antiqua bold",15),highlightthickness=2)
    password.place(x=300,y=190 )
    
    phoneno=Entry(R2,width=20,font=("Book Antiqua bold",15),highlightthickness=2)
    phoneno.place(x=300,y= 240 )
    
    address=Entry(R2,width=20,font=("Book Antiqua bold",15),highlightthickness=2)
    address.place(x=300,y= 290 )

    

    signUpbt = Button(R2,text = "SignUp",width=10,height=2,fg="black",font=('Book Antiqua',15,'bold'),justify='center',bg="aqua",command=Sigups)
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
                messagebox.showinfo("Sorry", "Wrong Password Entered")
  
    
    R3 =Toplevel()
    R3.geometry('900x600')
    R3.title("LOGIN NOW")
    
   
    

    lblInfo=Label(R3,text="Username",fg="gray",font=("Book Antiqua bold",15))
    lblInfo.place(x=230,y=200)
   
    lblInfo=Label(R3,text="Password",fg="gray",font=("Book Antiqua bold",15))
    lblInfo.place(x=230,y=250)

    e1= Entry(R3,width=18,font=("Book Antiqua bold",17),highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e1.place(x=330, y=190)

    e2= Entry(R3,width=18,font=("Book Antiqua bold",17),show="*",highlightthickness=2,bg="WHITE",relief=SUNKEN)
    e2.place(x=330, y=240)

    btn = Button(R3, text="LOGIN", width=10, height=2,fg="white",font=('Book Antiqua',15,'bold'),bg="orange",command=logininto)
    btn.place(x=480, y=300)
    
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
   
    Registerbt = Button(R4,text = "PREDICTION",width=17,height=2,font=('Book Antiqua',15,'bold'),justify='center',bg="light blue",command=prediction)
    
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
            
            import pickle
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
        lb1 = Label(window2, text="Chronic kidney prediction",font=('Book Antiqua',18,'bold'),justify='center',fg="green")
        lb1.place(x=40, y=70)
        lb1 = Label(window2, text="Age",font=('Book Antiqua',15,'bold'),fg="BLUE",anchor='w')
        lb1.place(x=100, y=120)
        E1=IntVar()
        e1= Entry(window2,width=10,textvariable=E1,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e1.place(x=300, y=120)

            
        lb2 = Label(window2, text="Bp",font=('Book Antiqua',15,'bold'),fg="BLUE",anchor='w')
        lb2.place(x=100, y=200)
        E2=DoubleVar()
        e2= Entry(window2,width=10,textvariable=E2,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e2.place(x=300, y=200)

        lb3 = Label(window2, text="Specific gravity",font=('Book Antiqua',15,'bold'),fg="BLUE",anchor='w')
        lb3.place(x=100, y=270)
        E3=IntVar()
        e3= Entry(window2,width=10,textvariable=E3,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e3.place(x=300, y=270) 

        lb4 = Label(window2, text="Albumin",font=('Book Antiqua',15,'bold'),fg="BLUE",anchor='w')
        lb4.place(x=100, y=350)
        E4=IntVar()
        e4= Entry(window2,width=10,textvariable=E4,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e4.place(x=300, y=350)

        lb5 = Label(window2, text="serum urate",font=('Book Antiqua',15,'bold'),fg="BLUE",anchor='w')
        lb5.place(x=100, y=430)
        E5=IntVar()
        e5= Entry(window2,width=10,textvariable=E5,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e5.place(x=300, y=430)
        btn1 = Button(window2, text= "PREDICTION", width=15, height=1,fg="white",font=('Book Antiqua',15,'bold'),bg="darkred",justify='center',command=svm1)
        btn1.place(x=230, y=510)
        
        lblIn=Label(window2,text="Accuracy",fg="black",font=("Book Antiqua",15))
        lblIn.place(x=100,y=550)

        lblIn=Label(window2,text="Status",fg="black",font=("Book Antiqua",15))
        lblIn.place(x=470,y=550)
        
        message =Label(window2, text="" ,bg="aqua"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message.place(x=40, y=590)
        
        message1 =Label(window2, text="" ,bg="aqua"  ,fg="red"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
        message1.place(x=350, y=590)
        bt = Button(window2,text = "Back",width=8,height=1,font=('Book Antiqua',15,'bold'),justify='center',bg="aquamarine",relief=SUNKEN,command=prediction)
        bt.place(x =0 ,y=2)
        
        
        
        #message1.configure(text= res)
        window2.mainloop()
        
       


#-----------------------decission tree----------------------------

    def decissiontree():
        R5.destroy()
        print('====================Decissiontree===================')
        
        def dt1():
            
            import pickle
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
        lb1 = Label(window3, text="Chronic kidney prediction",font=('Book Antiqua',15,'bold'),justify='center',fg="green")
        lb1.place(x=40, y=70)
        lb1 = Label(window3, text="Age",font=('Book Antiqua',15,'bold'),fg="grey",anchor='w')
        lb1.place(x=100, y=120)
        E1=IntVar()
        e1= Entry(window3,width=10,textvariable=E1,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e1.place(x=300, y=120)

            
        lb2 = Label(window3, text="Bp",font=('Book Antiqua',15,'bold'),fg="grey",anchor='w')
        lb2.place(x=100, y=200)
        E2=DoubleVar()
        e2= Entry(window3,width=10,textvariable=E2,font=("Book Antiqua",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e2.place(x=300, y=200)

        lb3 = Label(window3, text="Specific gravity",font=('Book Antiqua',15,'bold'),fg="grey",anchor='w')
        lb3.place(x=100, y=270)
        E3=IntVar()
        e3= Entry(window3,width=10,textvariable=E3,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e3.place(x=300, y=270) 

        lb4 = Label(window3, text="Albumin",font=('Book Antiqua',15,'bold'),fg="grey",anchor='w')
        lb4.place(x=100, y=350)
        E4=IntVar()
        e4= Entry(window3,width=10,textvariable=E4,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e4.place(x=300, y=350)

        lb5 = Label(window3, text="serum urate",font=('Book Antiqua',15,'bold'),fg="grey",anchor='w')
        lb5.place(x=100, y=430)
        E5=IntVar()
        e5= Entry(window3,width=10,textvariable=E5,font=("bold",15),highlightthickness=2,bg="WHITE",relief=SUNKEN)
        e5.place(x=300, y=430)
        btn1 = Button(window3, text= "PREDICTION", width=15, height=1,fg="white",font=('Book Antiqua',15,'bold'),bg="darkred",justify='center',command=dt1)
        btn1.place(x=230, y=510)
        
        lblIn=Label(window3,text="Accuracy",fg="black",font=("Book Antiqua",15))
        lblIn.place(x=100,y=550)

        lblIn=Label(window3,text="Status",fg="black",font=("Book Antiqua",15))
        lblIn.place(x=470,y=550)
        
        message =Label(window3, text="" ,bg="cyan"  ,fg="red"  ,width=30  ,height=2, activebackground = "dark blue" ,font=('Georgia', 15, ' bold ')) 
        message.place(x=40, y=590)
        
        message1 =Label(window3, text="" ,bg="cyan"  ,fg="red"  ,width=30  ,height=2, activebackground = "red" ,font=('Georgia', 15, ' bold ')) 
        message1.place(x=350, y=590)
        bt = Button(window3,text = "Back",width=8,height=1,font=('Book Antiqua',15,'bold'),justify='center',bg="aquamarine",relief=RAISED,command=prediction)
        bt.place(x =0 ,y=2)
        #bt1 = Button(window3,text = "refresh",width=8,height=1,font=('algerian',15,'bold'),justify='center',bg="aquamarine",relief=RAISED,command= )
        #bt1.place(x =150 ,y=2)
        
        #message1.configure(text= res)
        window3.mainloop()
        
    
    b1 = Button(R5,text ="SUPPORT VECTOR MACHINE" ,width=22,height=2,font=('Cambria',15,'bold'),justify='center',bg="turquoise",fg="white",relief=SUNKEN,command=svm)
    b1.place(x =200 ,y=100)
    
    b3 = Button(R5,text = "DECISION TREE",width=22,height=2,font=('Cambria',15,'bold'),justify='center',bg="turquoise",fg="white",relief=SUNKEN,command=decissiontree)
    b3.place(x =200 ,y=250)
    
    R5.mainloop()


main()




