import type {NextPage} from 'next'
import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'

export async function getStaticProps() {
    const helloResponse = await fetch('http://localhost:5000/')
        .then(response => response.json().then(json => json.lastmessage))
        .catch(error => error.toString())
    const predictResponse = await fetch('http://localhost:5000/predict')
        .then(response => response.json().then(json => JSON.stringify(json)))
        .catch(error => error.toString())
    return {
        props: {
            helloResponse,
            predictResponse,
        }
    }
}

export default function Home({helloResponse, predictResponse, explainResponse}) {

    return(
        <div className={styles.container}>
            <Head>
                <title>Create Next App</title>
                <meta name="description" content="Generated by create next app" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <main className={styles.main}>
                <h1 className={styles.title}>
                    Welcome.
                </h1>

                <p className={styles.description}>
                    Last Message: {' '}
                    <code className={styles.code}>{helloResponse}</code>
                </p>

                <p className={styles.description}>
                    Prediction: {' '}
                    <code className={styles.code}>{predictResponse}</code>
                </p>
                <p className={styles.description}>
                    Explanation:
                <Image

                    src="http://localhost:5000/explain"
                    height={300}
                    width={2000}
                    unoptimized={true}
                />
                </p>
            </main>

            <footer className={styles.footer}>
                <a
                    href="https://vercel.com?utm_source=create-next-app&utm_medium=default-template&utm_campaign=create-next-app"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    Powered by{' '}
                    <span className={styles.logo}>
            <Image src="/vercel.svg" alt="Vercel Logo" width={72} height={16} />
          </span>
                </a>
            </footer>
        </div>
    )
}