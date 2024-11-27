// @ts-check
import { themes as prismThemes } from 'prism-react-renderer';

const config = {
  title: 'BLADE ',
  tagline: 'Bayesian Log Normal Deconvolution',
  favicon: 'img/favicon.ico',
  url: 'https://aryamaanbose.github.io', // Your GitHub Pages URL
  baseUrl: '/BLADE_site/',               // Base URL for your project site
  organizationName: 'aryamaanbose',      // Your GitHub username
  projectName: 'BLADE_site',             // Your GitHub repository name

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'nl'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/aryamaanbose/BLADE_site/edit/main/BLADE/', // Link to edit docs in GitHub
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/aryamaanbose/BLADE_site/edit/main/BLADE/', // Link to edit blog posts in GitHub
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  themeConfig: {
    image: 'img/logo_final_small.png', // Image used for social media previews
    navbar: {
      title: 'BLADE v1.0.0',
      logo: {
        alt: 'BLADE Logo',
        src: 'img/logo_final_small.png',
      },
      items: [
        {
          type: 'doc',
          docId: 'installation',
          position: 'left',
          label: 'Installation',
        },
        {
          type: 'doc',
          docId: 'intro',
          position: 'left',
          label: 'Tutorial',
        },
        {
          type: 'doc',
          docId: 'about-us',
          position: 'left',
          label: 'About Us',
        },
        { to: '/blog', label: 'Blog', position: 'left' },
        {
          href: 'https://github.com/tgac-vumc/BLADE',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'r', 'bash'],
    },

    colorMode: {
      defaultMode: 'light',
      disableSwitch: true,
    },
  },
};

export default config;
