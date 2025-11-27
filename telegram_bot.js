const { Telegraf } = require("telegraf");
const TOKEN = '6255292930:AAH62cBEiJQJ3X7l-hiPwNFkS6pmegaIlSQ';
const bot = new Telegraf(TOKEN);
const Redis = require('ioredis');

const web_link = "https://botrecognizer.netlify.app";



bot.start((ctx) => {
  ctx.reply("Welcome :)))))", {
    reply_markup: {
      keyboard: [[{ text: "web app", web_app: { url: web_link } }]],
    },
  });
  
  const redisClient = Redis.createClient({
    host: 'localhost',
    port: 6379,
  });
  channel_name = 'my_channel'
  redisClient.subscribe(channel_name);
  redisClient.on('message', (channel,message) => {
    console.log('Received message:', message); // Add this line to inspect the message variable

    try {
      const parsedMessage = JSON.parse(message);
      const { text } = parsedMessage;
      console.log('Parsed message:', parsedMessage); // Add this line to inspect the parsed message
  
      ctx.reply(message);
    } catch (error) {
      console.error('Error parsing message:', error);
    }
    
  
  });
  
  


});



bot.launch();