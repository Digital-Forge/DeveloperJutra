import 'dotenv/config';
import { GoogleGenAI } from "@google/genai";
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import chalk from 'chalk';

const ai = new GoogleGenAI({});
const rl = readline.createInterface({ input, output });

// Ustawienia kolorów
const COLOR_USER = chalk.hex('#58a6ff');
const COLOR_AI = chalk.hex('#a371f7');
const COLOR_TOKENS = chalk.gray;
const COLOR_PROMPT = chalk.green('=> Wpisz swoje pytanie:');


async function startChatLoop() {
  console.log(chalk.bold('--- 🤖 Rozpoczęcie czatu z modelem Gemini 2.5 Flash ---'));
  console.log(chalk.yellow('Wpisz "exit", aby zakończyć rozmowę.'));
  console.log(' ');

  try {
    while (true) {
      const userInput = await rl.question(COLOR_PROMPT + ' ');

      if (userInput.toLowerCase() === 'exit') {
        console.log(chalk.bold('--- Zakończenie sesji czatu. Pa! ---'));
        rl.close();
        break;
      }

      console.log(COLOR_USER('-> Użytkownik: ') + userInput);

      const response = await ai.models.generateContent({
        model: "gemini-2.5-flash",
        contents: userInput,
      });

      const aiResponseText = response.text;
      console.log(COLOR_AI('-> Asystent: ') + aiResponseText);

      const userTokenCount = await ai.models.countTokens({
        model: "gemini-2.5-flash",
        contents: userInput,
      });

      const aiTokenCount = await ai.models.countTokens({
        model: "gemini-2.5-flash",
        contents: aiResponseText,
      });

      console.log(' ');
      console.log(COLOR_TOKENS('   [Tokens - Użytkownik: ' + userTokenCount.totalTokens + ' | Asystent: ' + aiTokenCount.totalTokens + ']'));
      console.log(' ');
    }

  } catch (error) {
    console.error(chalk.red('Wystąpił błąd w trakcie rozmowy:'), error.message);
    rl.close();
  }
}

startChatLoop();