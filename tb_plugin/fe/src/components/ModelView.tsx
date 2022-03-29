/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import CardHeader from '@material-ui/core/CardHeader'
import InputLabel from '@material-ui/core/InputLabel'
import MenuItem from '@material-ui/core/MenuItem'
import Grid from '@material-ui/core/Grid'
import Select, { SelectProps } from '@material-ui/core/Select'
import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'
import { Table } from 'antd'
import * as api from '../api'
import { TextListItem } from './TextListItem'

const useStyles = makeStyles((theme) => ({
  root: {
    flexGrow: 1
  },
  hide: {
    display: 'none'
  }
}))

export interface IProps {
  run: string
  worker: string
  span: string
}

const getKeyedTableColumns = (columns: api.KeyedColumn[]) => {
  return columns.map((col) => {
    return {
      dataIndex: col.key,
      key: col.key,
      title: col.name
    }
  })
}

const getTableRows = (rows: api.ModelStats[]) => {
  return rows.map((row) => {
    const data: any = {
      key: row.key,
      name: row.name,
      type: row.type,
      params: row.params,
      params_percentage: row.params_percentage,
      macs: row.macs,
      macs_percentage: row.macs_percentage,
      flops: row.flops,
      duration: row.duration,
      latency_percentage: row.latency_percentage,
      extra_repr: row.extra_repr
    }

    if (row.children.length) {
      data.children = getTableRows(row.children)
    }

    return data
  })
}

export const ModelView: React.FC<IProps> = (props) => {
  const { run, worker, span } = props
  const classes = useStyles()

  const [modelView, setModelView] = React.useState<
    api.ModelViewData | undefined
  >(undefined)

  const [columns, setColumns] = React.useState<any[]>([])
  const [rows, setRows] = React.useState<any[]>([])

  const [modules, setModules] = React.useState<number[]>([])
  const [module, setModule] = React.useState<number>(0)

  const [modelOverviews, setModelOverviews] = React.useState<
    api.ModelOverview[]
  >([])

  React.useEffect(() => {
    api.defaultApi
      .modelGet(run, worker, span)
      .then((resp) => {
        setModelView(resp)
        if (resp) {
          setModules(Array.from(Array(resp.data.length).keys()))
          // set the tree table data
          setColumns(getKeyedTableColumns(resp.columns))
          setRows(getTableRows(resp.data))
          setModelOverviews(resp.overview)
        }
      })
      .catch((e) => {
        if (e.status == 404) {
          setModules([])
          setRows([])
        }
      })
  }, [run, worker, span])

  // const handleModuleChange: SelectProps['onChange'] = (event) => {
  //   setModule(event.target.value as number)
  // }

  // const moduleComponent = () => {
  //   const moduleFragment = (
  //     <React.Fragment>
  //       <InputLabel id="module-graph">Module</InputLabel>
  //       <Select value={module} onChange={handleModuleChange}>
  //         {modules.map((m) => (
  //           <MenuItem value={m}>{m}</MenuItem>
  //         ))}
  //       </Select>
  //     </React.Fragment>
  //   )

  //   if (!modules || modules.length <= 1) {
  //     return <div className={classes.hide}>{moduleFragment}</div>
  //   } else {
  //     return moduleFragment
  //   }
  // }

  return (
    <div className={classes.root}>
      <Card variant="outlined">
        <CardHeader title="Model Property" />
        <CardContent>
          <Grid container item spacing={1}>
            <Grid item sm={12}>
              {React.useMemo(
                () => (
                  <Card variant="outlined">
                    <CardHeader title="Summary" />
                    <CardContent>
                      {modelOverviews.map((item) => (
                        <TextListItem name={item.title} value={item.value} />
                      ))}
                    </CardContent>
                  </Card>
                ),
                [modelOverviews]
              )}
            </Grid>
          </Grid>
          <Grid item sm={12}>
            {rows && rows.length > 0 && (
              <Table
                size="small"
                bordered
                columns={columns}
                dataSource={rows}
                expandable={{
                  defaultExpandAllRows: false
                }}
              />
            )}
          </Grid>
        </CardContent>
      </Card>
    </div>
  )
}
