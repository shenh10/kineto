/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import Grid from '@material-ui/core/Grid'
import { makeStyles } from '@material-ui/core/styles'
import CardMedia from '@material-ui/core/CardMedia'
import * as React from 'react'
import * as api from '../api'

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  }
}))

export interface IProps {
  run: string
  worker: string
  span: string
}

export const CodebaseView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const [pythonBottleneck, setPythonBottleneck] = React.useState<
    api.PythonBottleneck | undefined
  >(undefined)

  React.useEffect(() => {
    api.defaultApi.codebaseGet(run, worker, span).then((resp) => {
      setPythonBottleneck(resp.python_bottleneck)
    })
  }, [run, worker, span])
  const classes = useStyles()

  return (
    <div className={classes.root}>
      <Grid container spacing={1}>
        <Grid container item>
          {pythonBottleneck && (
            <Grid item sm={12}>
              <Card variant="outlined">
                <CardHeader title="Python Bottleneck Analaysis" />
                <CardContent>
                  <CardMedia
                    component="img"
                    src={`data:image/png;base64, ${pythonBottleneck.image_content}`}
                    alt="python profiling graph"
                  />
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </Grid>
    </div>
  )
}
