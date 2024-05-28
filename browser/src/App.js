import './App.css';
import React, { useState, useEffect } from 'react';
import {Helmet} from "react-helmet";

const ws = new WebSocket('ws://localhost:8001');

function loadingIndicator(){
    return (
      <div style={{textAlign: 'right'}} >
        <div class="spinner"></div>
      </div>
    )
}


function App() {
  const [messages, setMessages] = useState([]);
  const [tempMessage, setTempMessage] = useState('');
  const [newMessage, setNewMessage] = useState('');
  const [waitingForServer, setWaitingForServer] = useState(true);
  const [waitingForModelResponse, setWaitingForModelResponse] = useState(true);
  const [error, setError] = useState("");

  function removeSystemReadyMessages(messages) {
    return messages.filter(message => !(message.role == 'system' || message.content == 'Ready'));
  }


  useEffect(() => {
    ws.onmessage = event => {
      const message = JSON.parse(event.data);
      console.log(message);
      if (message.role === 'system'){
        if(message.content === 'Ready'){
          setWaitingForServer(false);
          return;
        }
      }


      if (message.message_curr !== undefined){
        setTempMessage(message.message_curr);
        return;
      }

      if (message.message_hist){
        var new_hist = removeSystemReadyMessages(message.message_hist);

        if (new_hist.length < message.message_hist.length){
          setWaitingForServer(false);
        }
        setMessages(new_hist);
        return;
      }

      if (message.error) {
        setError(message.error);
        return;
      }

      setTempMessage(tempMessage + message.content)

      if (message.ended){
        setMessages([...messages, { role: message.role, content: tempMessage, timestamp: Date.now() } ]);
        setTempMessage('');
      }

    };

    ws.onclose = event => {
      setError("The server closed the connection, please refresh the page and try again.");
    }

    return () => {};
  }, [messages, tempMessage]);

  const handleSendMessage = () => {
    var formated_message = { role: 'user', content: newMessage, timestamp: Date.now() }
    // Add new message to list of messages
    setMessages([...messages, { role: 'user', content: newMessage, timestamp: Date.now() }]);
    // Send message over WebSocket
    ws.send(JSON.stringify(formated_message));
    // Clear input field
    setNewMessage('');

    var chat_window = document.getElementById("chat-window");
    chat_window.scrollTop = chat_window.scrollHeight;
  };

  if (error !== "") {
    return (
      <div>
        <p onLoad={setTimeout(function(){ window.location.reload(); }, 5000)}></p>
         <h1> (1/2) Please wait for a connection to be established....</h1>
      </div>
    )
  }

  if (waitingForServer) {
    return (
      <div>
        <h1> (2/2) Please wait the system is being setup, you may need to enter some prompts on the terminal to boot the application.</h1>
      </div>
    )
  }


  var pending_message = (<div></div>)

  if (tempMessage !== ''){
    pending_message = (<div class={"message model pending"}>{tempMessage}...{loadingIndicator()}</div>)
  }

  return (
    <div id="chat-background">
      <div id="chat-window">
        <ul>
          {messages.map((message, index) => (
            <div class={"message " + message.role} key={index}>{message.content}</div>
          ))}
          {pending_message}
        </ul>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <br/>
        <div class="input-div">
          <form id="input-container" class="fixed-bottom" onSubmit={event => {
            event.preventDefault();
            handleSendMessage();

          }}>
            <input id="input-box" type="text" value={newMessage} onChange={event => setNewMessage(event.target.value)} />
            <button  id="send-btn" type="submit" disabled={tempMessage !== ''}>Ask TorchChat</button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default App;
