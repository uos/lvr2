
#ifndef QDEBUGSTREAM_HPP_
#define QDEBUGSTREAM_HPP_


#include <QObject>
#include <QPlainTextEdit>
#include <QMutex>

#include <iostream>

///////////////////////////////////////////////////////////////////////////
//
class QDebugStream : public QObject, std::basic_streambuf<char>{

   Q_OBJECT

Q_SIGNALS:
   void sendString(QString text);

public:
   QDebugStream(std::ostream &stream, QPlainTextEdit* text_edit) : m_stream(stream){
      log_window = text_edit;
      m_old_buf = stream.rdbuf();
      stream.rdbuf(this);

      connect(this, SIGNAL(sendString(QString)), text_edit, SLOT(appendPlainText (QString)));
   }
   ~QDebugStream(){
   // output anything that is left
   if (!m_string.empty())
      Q_EMIT sendString(m_string.c_str());
      //log_window->appendPlainText(m_string.c_str());
      m_stream.rdbuf(m_old_buf);
   }

protected:
   virtual int_type overflow(int_type v){
      mutex.lock();
      if (v == '\n'){
         Q_EMIT sendString(m_string.c_str());
         //log_window->appendPlainText(m_string.c_str());
         m_string.erase(m_string.begin(), m_string.end());
      }
      else
         m_string += v;

      mutex.unlock();
      return v;
   }

   virtual std::streamsize xsputn(const char *p, std::streamsize n){

      mutex.lock();

      m_string.append(p, p + n);
      int pos = 0;
      while (pos != std::string::npos){
          pos = m_string.find('\n');
          if (pos != std::string::npos){
              std::string tmp(m_string.begin(), m_string.begin() + pos);
              Q_EMIT sendString(tmp.c_str());
              //log_window->appendPlainText(tmp.c_str());
              m_string.erase(m_string.begin(), m_string.begin() + pos + 1);
          }
      }

      mutex.unlock();
      return n;
   }

private:
   std::ostream &m_stream;
   std::streambuf *m_old_buf;
   std::string m_string;
   QPlainTextEdit* log_window;
   QMutex mutex;
};


#endif /* QDEBUGSTREAM_HPP_ */
