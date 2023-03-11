import "./globals.css";

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <title>Serverless Inference</title>
        <meta name="description" content="Serverless Inference" />
        <link rel="icon" href="/favicon.ico" />
      </head>
      <body>{children}</body>
    </html>
  );
}
