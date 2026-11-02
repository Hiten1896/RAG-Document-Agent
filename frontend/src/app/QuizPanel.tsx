'use client';

import { useState } from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Loader2,
  RotateCcw,
  X,
  XCircle,
} from 'lucide-react';
import type { QuizDifficulty, QuizResponse } from '@/lib/api';

interface QuizPanelProps {
  data: QuizResponse | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onGenerate: (count: number, difficulty: QuizDifficulty) => void;
}

function answersMatch(left: string | undefined, right: string): boolean {
  return left?.trim().toLowerCase() === right.trim().toLowerCase();
}

export default function QuizPanel({
  data,
  loading,
  error,
  onClose,
  onGenerate,
}: QuizPanelProps) {
  const [count, setCount] = useState(5);
  const [difficulty, setDifficulty] = useState<QuizDifficulty>('medium');
  const [mode, setMode] = useState<'quiz' | 'flashcards'>('quiz');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [answers, setAnswers] = useState<Record<number, string>>({});
  const [visibleSources, setVisibleSources] = useState<Record<number, boolean>>({});
  const [flipped, setFlipped] = useState(false);

  const questions = data?.questions ?? [];
  const currentQuestion = questions[currentIndex];
  const selectedAnswer = answers[currentIndex];
  const correctCount = questions.reduce(
    (total, question, index) =>
      total + Number(answersMatch(answers[index], question.correct_answer)),
    0,
  );
  const answeredCount = Object.keys(answers).length;
  const quizComplete = questions.length > 0 && answeredCount === questions.length;

  const selectAnswer = (answer: string) => {
    if (selectedAnswer !== undefined) return;
    setAnswers((previous) => ({ ...previous, [currentIndex]: answer }));
  };

  const toggleSource = () => {
    setVisibleSources((previous) => ({
      ...previous,
      [currentIndex]: !previous[currentIndex],
    }));
  };

  const changeQuestion = (nextIndex: number) => {
    setCurrentIndex(nextIndex);
    setFlipped(false);
  };

  return (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/60 p-3 sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="quiz-panel-title"
    >
      <section className="flex max-h-[92vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900">
        <header className="flex items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-slate-800 sm:px-6">
          <div>
            <h2 id="quiz-panel-title" className="text-base font-semibold text-slate-900 dark:text-slate-100">
              Quiz me
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Questions and answers are checked against your document.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close quiz"
            className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-800 dark:hover:bg-slate-800 dark:hover:text-slate-100"
          >
            <X className="h-4 w-4" />
          </button>
        </header>

        <div className="flex-1 space-y-4 overflow-y-auto p-4 sm:p-6">
          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1 text-xs font-medium text-slate-600 dark:text-slate-300">
              Questions
              <select
                value={count}
                onChange={(event) => setCount(Number(event.target.value))}
                className="rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-sm text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              >
                {Array.from({ length: 10 }, (_, index) => index + 1).map((value) => (
                  <option key={value} value={value}>{value}</option>
                ))}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs font-medium text-slate-600 dark:text-slate-300">
              Difficulty
              <select
                value={difficulty}
                onChange={(event) => {
                  const value = event.target.value;
                  if (value === 'easy' || value === 'medium' || value === 'hard') {
                    setDifficulty(value);
                  }
                }}
                className="rounded-lg border border-slate-300 bg-white px-2.5 py-2 text-sm capitalize text-slate-800 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
              >
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </label>
            <button
              type="button"
              onClick={() => onGenerate(count, difficulty)}
              disabled={loading}
              className="inline-flex min-h-10 items-center gap-2 rounded-lg bg-indigo-600 px-4 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-wait disabled:opacity-60"
            >
              {loading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RotateCcw className="h-4 w-4" />
              )}
              {data ? 'Generate new quiz' : 'Generate quiz'}
            </button>
          </div>

          {error && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-900/70 dark:bg-rose-950/30 dark:text-rose-300"
            >
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {loading && (
            <div className="flex min-h-40 flex-col items-center justify-center gap-3 text-sm text-slate-500 dark:text-slate-400">
              <Loader2 className="h-6 w-6 animate-spin text-indigo-500" />
              Building a grounded quiz…
            </div>
          )}

          {!loading && questions.length === 0 && !error && (
            <div className="rounded-xl border border-dashed border-slate-300 px-5 py-10 text-center text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
              Choose the number and difficulty, then generate a quiz from the active chat&apos;s document.
            </div>
          )}

          {!loading && currentQuestion && (
            <>
              <div className="flex items-center justify-between gap-3">
                <div className="flex rounded-lg bg-slate-100 p-1 dark:bg-slate-800">
                  {(['quiz', 'flashcards'] as const).map((tab) => (
                    <button
                      key={tab}
                      type="button"
                      onClick={() => {
                        setMode(tab);
                        setFlipped(false);
                      }}
                      className={`rounded-md px-3 py-1.5 text-xs font-medium capitalize transition ${
                        mode === tab
                          ? 'bg-white text-indigo-700 shadow-sm dark:bg-slate-700 dark:text-indigo-200'
                          : 'text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100'
                      }`}
                    >
                      {tab}
                    </button>
                  ))}
                </div>
                <span className="text-xs text-slate-500 dark:text-slate-400">
                  {mode === 'quiz'
                    ? `Score ${correctCount}/${answeredCount} · `
                    : ''}
                  {currentIndex + 1} of {questions.length}
                </span>
              </div>

              {mode === 'quiz' ? (
                <div className="rounded-xl border border-slate-200 p-4 dark:border-slate-700 sm:p-5">
                  <h3 className="mb-4 text-base font-semibold leading-relaxed text-slate-900 dark:text-slate-100">
                    {currentQuestion.question}
                  </h3>
                  <div className="space-y-2">
                    {currentQuestion.options.map((option, optionIndex) => {
                      const isCorrect = answersMatch(
                        option,
                        currentQuestion.correct_answer,
                      );
                      const isSelected = option === selectedAnswer;
                      const stateClass =
                        selectedAnswer === undefined
                          ? 'border-slate-200 text-slate-800 hover:border-indigo-400 hover:bg-indigo-50 dark:border-slate-700 dark:text-slate-100 dark:hover:border-indigo-500 dark:hover:bg-indigo-950/30'
                          : isCorrect
                            ? 'border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-200'
                            : isSelected
                              ? 'border-rose-300 bg-rose-50 text-rose-800 dark:border-rose-800 dark:bg-rose-950/30 dark:text-rose-200'
                              : 'border-slate-200 text-slate-600 opacity-70 dark:border-slate-700 dark:text-slate-400';
                      return (
                        <button
                          key={`${optionIndex}-${option}`}
                          type="button"
                          onClick={() => selectAnswer(option)}
                          disabled={selectedAnswer !== undefined}
                          className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2.5 text-left text-sm transition disabled:cursor-default ${stateClass}`}
                        >
                          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-semibold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                            {String.fromCharCode(65 + optionIndex)}
                          </span>
                          <span className="flex-1">{option}</span>
                          {selectedAnswer !== undefined && isCorrect && (
                            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-300" />
                          )}
                          {selectedAnswer !== undefined && isSelected && !isCorrect && (
                            <XCircle className="h-4 w-4 shrink-0 text-rose-600 dark:text-rose-300" />
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {selectedAnswer !== undefined && (
                    <div className="mt-4 rounded-lg bg-slate-50 p-3 text-sm dark:bg-slate-800/70">
                      <p className="font-semibold text-slate-800 dark:text-slate-100">
                        {answersMatch(
                          selectedAnswer,
                          currentQuestion.correct_answer,
                        )
                          ? 'Correct!'
                          : `Correct answer: ${currentQuestion.correct_answer}`}
                      </p>
                      <p className="mt-1 text-slate-600 dark:text-slate-300">{currentQuestion.explanation}</p>
                      <button
                        type="button"
                        onClick={toggleSource}
                        className="mt-2 text-xs font-medium text-indigo-700 underline underline-offset-2 dark:text-indigo-300"
                      >
                        {visibleSources[currentIndex] ? 'Hide source' : 'Show source'}
                      </button>
                      {visibleSources[currentIndex] && (
                        <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{currentQuestion.source}</p>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => setFlipped((previous) => !previous)}
                  aria-label={flipped ? 'Show flashcard question' : 'Reveal flashcard answer'}
                  className="flex min-h-64 w-full flex-col items-center justify-center rounded-xl border border-indigo-200 bg-indigo-50 p-6 text-center transition hover:border-indigo-400 dark:border-indigo-900/70 dark:bg-indigo-950/30 sm:min-h-72"
                >
                  <span className="mb-3 text-[11px] font-semibold uppercase tracking-wider text-indigo-600 dark:text-indigo-300">
                    {flipped ? 'Answer' : 'Question'} · tap to flip
                  </span>
                  <span className="text-lg font-semibold leading-relaxed text-slate-900 dark:text-slate-100">
                    {flipped ? currentQuestion.correct_answer : currentQuestion.question}
                  </span>
                  {flipped && (
                    <span className="mt-3 max-w-lg text-sm text-slate-600 dark:text-slate-300">
                      {currentQuestion.explanation}
                    </span>
                  )}
                  {flipped && visibleSources[currentIndex] && (
                    <span className="mt-4 text-xs text-slate-500 dark:text-slate-400">
                      Source: {currentQuestion.source}
                    </span>
                  )}
                </button>
              )}

              <div className="flex items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={() => changeQuestion(Math.max(0, currentIndex - 1))}
                  disabled={currentIndex === 0}
                  className="inline-flex min-h-10 items-center gap-1 rounded-lg px-3 text-sm font-medium text-slate-600 hover:bg-slate-100 disabled:opacity-40 dark:text-slate-300 dark:hover:bg-slate-800"
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous
                </button>
                {mode === 'flashcards' && flipped && (
                  <button
                    type="button"
                    onClick={toggleSource}
                    className="text-xs font-medium text-indigo-700 underline underline-offset-2 dark:text-indigo-300"
                  >
                    {visibleSources[currentIndex] ? 'Hide source' : 'Show source'}
                  </button>
                )}
                {mode === 'quiz' && quizComplete && currentIndex === questions.length - 1 ? (
                  <span role="status" className="text-sm font-semibold text-emerald-700 dark:text-emerald-300">
                    Final score: {correctCount}/{questions.length}
                  </span>
                ) : (
                  <button
                    type="button"
                    onClick={() => changeQuestion(Math.min(questions.length - 1, currentIndex + 1))}
                    disabled={currentIndex === questions.length - 1 || (mode === 'quiz' && selectedAnswer === undefined)}
                    className="inline-flex min-h-10 items-center gap-1 rounded-lg bg-slate-100 px-3 text-sm font-medium text-slate-700 transition hover:bg-slate-200 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
                  >
                    Next
                    <ChevronRight className="h-4 w-4" />
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      </section>
    </div>
  );
}
