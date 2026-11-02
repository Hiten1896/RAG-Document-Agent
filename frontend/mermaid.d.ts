declare module 'mermaid/dist/mermaid.js' {
  const mermaid: {
    initialize(options: { startOnLoad: boolean; securityLevel: 'strict'; theme: string }): void;
    render(id: string, definition: string): Promise<{ svg: string }>;
  };
  export default mermaid;
}
