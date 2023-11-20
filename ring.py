from smtplib import SMTP_SSL
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header


host_server = 'smtp.pku.edu.cn'  #smtp服务器
sender = '1900013009@pku.edu.cn' #发件人邮箱
pwd = 'EHVOcdzi4162'
receiver = '1900013009@pku.edu.cn'#收件人邮箱
mail_title = 'new version' #邮件标题
mail_content = "new version" #邮件正文内容
# 初始化一个邮件主体
msg = MIMEMultipart()
msg["Subject"] = Header(mail_title,'utf-8')
msg["From"] = sender
# msg["To"] = Header("测试邮箱",'utf-8')
msg['To'] = ";".join(receiver)
# 邮件正文内容
msg.attach(MIMEText(mail_content,'plain','utf-8'))



smtp = SMTP_SSL(host_server) # ssl登录

smtp.login(sender,pwd)

smtp.sendmail(sender,receiver,msg.as_string())

smtp.quit()
